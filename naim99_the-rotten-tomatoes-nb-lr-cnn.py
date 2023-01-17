#https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

from multiprocessing import Pool

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#libraries to import ! you should read about every one og them ! 



import pandas as pd

import numpy as np

import re

import numpy as np

import pandas as pd

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

from textblob import TextBlob

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble



import pandas, xgboost, numpy, textblob, string

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report

from sklearn.model_selection import train_test_split



from warnings import filterwarnings

filterwarnings('ignore')
train_data = pd.read_csv('../input/moviereviewsentimentanalysiskernelsonly/train.tsv/train.tsv', sep="\t")

test_data = pd.read_csv('../input/moviereviewsentimentanalysiskernelsonly/test.tsv/test.tsv', sep="\t")

sub = pd.read_csv('../input/moviereviewsentimentanalysiskernelsonly/sampleSubmission.csv', sep=",")
train = pd.read_csv('../input/moviereviewsentimentanalysiskernelsonly/train.tsv/train.tsv', sep="\t")

test = pd.read_csv('../input/moviereviewsentimentanalysiskernelsonly/test.tsv/test.tsv', sep="\t")

sub = pd.read_csv('../input/moviereviewsentimentanalysiskernelsonly/sampleSubmission.csv', sep=",")
train_data.rename(columns={'Phrase':'text' , 'Sentiment':'target'}, inplace=True)

test_data.rename(columns={'Phrase':'text'}, inplace=True)
train_data['text'][1]
import re

def  clean_text(df, text_field, new_text_field_name):

    df[new_text_field_name] = df[text_field].str.lower() #lowercase

    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  

    # remove numbers

    #remove.............. (#re sub / search/ ..)

    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))

    

    return df

#read about dataframes ! TABLEAU !! 

#data_clean : new dataframe

data_clean = clean_text(train_data, 'text', 'text_clean')

data_clean_test = clean_text(test_data,'text', 'text_clean')

data_clean.head()
import nltk.corpus

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')

stop #the list of the stopwords ! 
data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

data_clean.head()
#Tokenization : word_tokenize ! 

import nltk 

nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

data_clean['text_tokens'] = data_clean['text_clean'].apply(lambda x: word_tokenize(x))

data_clean.head()
#stemming #PorterStemmer 

from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize

def word_stemmer(text):

    stem_text = [PorterStemmer().stem(i) for i in text]

    return stem_text

data_clean['text_clean_tokens'] = data_clean['text_tokens'].apply(lambda x: word_stemmer(x))

data_clean.head()
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

def word_lemmatizer(text):

    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]

    return lem_text

data_clean['text_clean_tokens'] = data_clean['text_tokens'].apply(lambda x: word_lemmatizer(x))

data_clean.head()
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)







data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: remove_URL(x))
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: remove_html(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: remove_emoji(x))
import string

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: remove_punct(x))
freq = pd.Series(' '.join(data_clean['text_clean']).split()).value_counts()[:10]



freq = list(freq.index)

data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
X_train, X_test, Y_train, Y_test = train_test_split(data_clean['text_clean'], 

                   

                                                    data_clean['target'], 

                                                    test_size = 0.2,

                                                    random_state = 10)



#HYPERPARAMETERS ! 

#SPLITTING THE DATA ! 


tfidf = TfidfVectorizer(encoding='utf-8',

                       ngram_range=(1,3),

                       max_df=1.0,

                       min_df=10,

                       max_features=500,

                       norm='l2',

                       sublinear_tf=True)
train_features = tfidf.fit_transform(X_train).toarray()

print(train_features.shape)
test_features = tfidf.transform(X_test).toarray()

print(test_features.shape)
train_labels = Y_train

test_labels = Y_test


import pandas as pd

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
mnb_classifier = MultinomialNB()
mnb_classifier.fit(train_features, train_labels)
mnb_prediction = mnb_classifier.predict(test_features)
training_accuracy = accuracy_score(train_labels, mnb_classifier.predict(train_features))

print(training_accuracy)
testing_accuracy = accuracy_score(test_labels, mnb_prediction)

print(testing_accuracy)
print(classification_report(test_labels, mnb_prediction))
conf_matrix = confusion_matrix(test_labels, mnb_prediction)

print(conf_matrix)
loj = linear_model.LogisticRegression()

loj_model = loj.fit(train_features, train_labels)

y_pred = loj_model.predict(test_features)





accuracy_score(test_labels,y_pred)
test_vectorizer =tfidf.transform( data_clean_test['text_clean']).toarray()
test_vectorizer.shape
final_predictions = mnb_classifier.predict(test_vectorizer)
final_predictions
submission_df = pd.DataFrame()
submission_df['PhraseId'] = data_clean_test['PhraseId']

submission_df['target'] = final_predictions
submission_df
submission_df['target'].value_counts()
submission = submission_df.to_csv('Result.csv',index = False)
seed = 0



import random

import numpy as np

import tensorflow as tf

tf.random.set_seed(seed) 
import pandas as pd



train = pd.read_csv('../input/moviereviewsentimentanalysiskernelsonly/train.tsv/train.tsv',  sep="\t")

test = pd.read_csv('../input/moviereviewsentimentanalysiskernelsonly/test.tsv/test.tsv',  sep="\t")
train.head()
train['Sentiment'].value_counts()
def format_data(train, test, max_features, maxlen):

    """

    Convert data to proper format.

    1) Shuffle

    2) Lowercase

    3) Sentiments to Categorical

    4) Tokenize and Fit

    5) Convert to sequence (format accepted by the network)

    6) Pad

    7) Voila!

    """

    from keras.preprocessing.text import Tokenizer

    from keras.preprocessing.sequence import pad_sequences

    from keras.utils import to_categorical

    

    train = train.sample(frac=1).reset_index(drop=True)

    train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())

    test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())



    X = train['Phrase']

    test_X = test['Phrase']

    Y = to_categorical(train['Sentiment'].values)



    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(X))



    X = tokenizer.texts_to_sequences(X)

    X = pad_sequences(X, maxlen=maxlen)

    test_X = tokenizer.texts_to_sequences(test_X)

    test_X = pad_sequences(test_X, maxlen=maxlen)



    return X, Y, test_X

maxlen = 125

max_features = 15000



X, Y, test_X = format_data(train, test, max_features, maxlen)
X
Y
test_X
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=seed)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, BatchNormalization

from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten

from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam

from tensorflow.keras import Sequential



from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping



model = Sequential()



# Input / Embdedding

model.add(Embedding(max_features, 150, input_length=maxlen))



# CNN

model.add(SpatialDropout1D(0.2))



model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))



model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))



model.add(Flatten())



# Output layer

model.add(Dense(5, activation='sigmoid'))
epochs = 5

batch_size = 32
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=1)
sub = pd.read_csv('../input/moviereviewsentimentanalysiskernelsonly/sampleSubmission.csv')



sub['Sentiment'] = model.predict_classes(test_X, batch_size=batch_size, verbose=1)

sub.to_csv('sub_cnn.csv', index=False)