# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow.keras.layers as L

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.optimizers import Adam



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import plotly.figure_factory as ff

import seaborn as sns





import numpy as np 

import pandas as pd



import nltk

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import TweetTokenizer

from nltk.tokenize import word_tokenize 

from nltk.corpus import stopwords



import re
train = pd.read_csv('../input/covid-19-nlp-text-classification/Corona_NLP_train.csv',encoding='latin1')

test = pd.read_csv('../input/covid-19-nlp-text-classification/Corona_NLP_test.csv',encoding='latin1')



train.head()
print('Examples in train data: {}'.format(len(train)))

print('Examples in test data: {}'.format(len(test)))
train.isna().sum()
dist_train = train['Sentiment'].value_counts()

dist_test = test['Sentiment'].value_counts()



def ditribution_plot(x,y,name):

    fig = go.Figure([

        go.Bar(x=x, y=y)

    ])



    fig.update_layout(title_text=name)

    fig.show()
ditribution_plot(x= dist_train.index, y= dist_train.values, name= 'Class Distribution train')
ditribution_plot(x= dist_test.index, y= dist_test.values, name= 'Class Distribution test')
X = train['OriginalTweet'].copy()

y = train['Sentiment'].copy()
def data_cleaner(tweet):

    

    # remove urls

    tweet = re.sub(r'http\S+', ' ', tweet)

    

    # remove html tags

    tweet = re.sub(r'<.*?>',' ', tweet)

    

    # remove digits

    tweet = re.sub(r'\d+',' ', tweet)

    

    # remove hashtags

    tweet = re.sub(r'#\w+',' ', tweet)

    

    # remove mentions

    tweet = re.sub(r'@\w+',' ', tweet)

    

    #removing stop words

    tweet = tweet.split()

    tweet = " ".join([word for word in tweet if not word in stop_words])

    

    return tweet





stop_words = stopwords.words('english')



X_cleaned = X.apply(data_cleaner)

X_cleaned.head()
tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_cleaned)



X = tokenizer.texts_to_sequences(X_cleaned)



vocab_size = len(tokenizer.word_index)+1



print("Vocabulary size: {}".format(vocab_size))

print("\nExample:\n")

print("Sentence:\n{}".format(X_cleaned[6]))

print("\nAfter tokenizing :\n{}".format(X[6]))



X = pad_sequences(X, padding='post')

print("\nAfter padding :\n{}".format(X[6]))
encoding = {'Extremely Negative': 0,

            'Negative': 0,

            'Neutral': 1,

            'Positive':2,

            'Extremely Positive': 2

           }



labels = ['Negative', 'Neutral', 'Positive']

           



y.replace(encoding, inplace=True)
tf.keras.backend.clear_session()



# hyper parameters

EPOCHS = 2

BATCH_SIZE = 32

embedding_dim = 16

units = 256



model = tf.keras.Sequential([

    L.Embedding(vocab_size, embedding_dim, input_length=X.shape[1]),

    L.Bidirectional(L.LSTM(units,return_sequences=True)),

    L.GlobalMaxPool1D(),

    L.Dropout(0.4),

    L.Dense(64, activation="relu"),

    L.Dropout(0.4),

    L.Dense(3)

])





model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),

              optimizer='adam',metrics=['accuracy']

             )



model.summary()
history = model.fit(X, y, epochs=EPOCHS, validation_split=0.12, batch_size=BATCH_SIZE)
fig = px.line(

    history.history, y=['accuracy', 'val_accuracy'],

    labels={'index': 'epoch', 'value': 'accuracy'}

)



fig.show()
fig = px.line(

    history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'loss'}

)



fig.show()
X_test = test['OriginalTweet'].copy()

y_test = test['Sentiment'].copy()



X_test = X_test.apply(data_cleaner)



X_test = tokenizer.texts_to_sequences(X_test)



X_test = pad_sequences(X_test, padding='post')



y_test.replace(encoding, inplace=True)
pred = model.predict_classes(X_test)
loss, acc = model.evaluate(X_test,y_test,verbose=0)

print('Test loss: {}'.format(loss))

print('Test Accuracy: {}'.format(acc))
conf = confusion_matrix(y_test, pred)



cm = pd.DataFrame(

    conf, index = [i for i in labels],

    columns = [i for i in labels]

)



plt.figure(figsize = (12,7))

sns.heatmap(cm, annot=True, fmt="d")

plt.show()
print(classification_report(y_test, pred, target_names=labels))