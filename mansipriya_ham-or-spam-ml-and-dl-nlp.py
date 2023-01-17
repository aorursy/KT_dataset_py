# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

import nltk

from nltk.corpus import stopwords

import string



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")
data.head()
data.describe()
count = data['Category'].value_counts()

print(count)
print("Spam % is ",(count[1]/float(count[0]+count[1]))*100)
data.drop_duplicates(inplace = True)
spam_list = data[data["Category"] == "spam"]["Message"].unique().tolist()

spam = " ".join(spam_list)

spam_wordcloud = WordCloud().generate(spam)

plt.figure(figsize=(12,8))

plt.imshow(spam_wordcloud)

plt.show()
ham_list = data[data["Category"] == "ham"]["Message"].unique().tolist()

ham = " ".join(ham_list)

ham_wordcloud = WordCloud().generate(ham)

plt.figure(figsize=(12,8))

plt.imshow(ham_wordcloud)

plt.show()
# mapping labels to 1 and 0

data['Category'] = data.Category.map({'ham':0, 'spam':1})
X=data['Message']

y=data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=201)
# vectorising the text

vect = CountVectorizer()

vect.fit(X_train)
vect.vocabulary_
# transform

X_train_transformed = vect.transform(X_train)

X_test_tranformed =vect.transform(X_test)
from sklearn.naive_bayes import BernoulliNB



# instantiate bernoulli NB object

bnb = BernoulliNB()



# fit 

bnb.fit(X_train_transformed,y_train)



# predict class

y_pred_class = bnb.predict(X_test_tranformed)



# predict probability

y_pred_proba =bnb.predict_proba(X_test_tranformed)



# accuracy

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)
metrics.confusion_matrix(y_test, y_pred_class)
#stop words are useless words

nltk.download('stopwords')
#Tokenization (a list of tokens), will be used as the analyzer

#1.Punctuations are [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]

#2.Stop words in natural language processing, are useless words (data).

def process_text(text):

    

    #1 Remove Punctuationa

    nopunc = [char for char in text if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    

    #2 Remove Stop Words

    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    

    #3 Return a list of clean words

    return clean_words
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(data['Message'])
X_train, X_test, y_train, y_test = train_test_split(messages_bow, data['Category'], random_state = 201)
# instantiate bernoulli NB object

bnb = BernoulliNB()



# fit 

bnb.fit(X_train,y_train)



# predict class

y_pred_class = bnb.predict(X_test)



# predict probability

y_pred_proba =bnb.predict_proba(X_test)



# accuracy

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



Spam_model = LogisticRegression(solver='liblinear', penalty='l1')

Spam_model.fit(X_train, y_train)

pred = Spam_model.predict(X_test)

accuracy_score(y_test,pred)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



parameters_KNN = {'n_neighbors': (5,10,15,20), }



grid_KNN = GridSearchCV( KNeighborsClassifier(), parameters_KNN, cv=5,

                        n_jobs=-1, verbose=1)



grid_KNN.fit(X_train, y_train)
print(grid_KNN.best_params_)

print(grid_KNN.best_score_)
#trying diffferent values for n_neighbours

parameters_KNN = {'n_neighbors': (4,5,6,7), }



grid_KNN = GridSearchCV( KNeighborsClassifier(), parameters_KNN, cv=5,

                        n_jobs=-1, verbose=1)



grid_KNN.fit(X_train, y_train)
print(grid_KNN.best_params_)

print(grid_KNN.best_score_)
model = KNeighborsClassifier(n_neighbors=5)



# Train the model using the training sets

model.fit(X_train,y_train)



#Predict Output

pred= model.predict(X_test)

accuracy_score(y_test,pred)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU,SimpleRNN

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping





import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff
try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
data
X_train, X_test, y_train, y_test = train_test_split(data.Message.values, data.Category.values, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)
data['Message'].apply(lambda x:len(str(x).split())).max()
# auc value returned 

def roc_auc(predictions,target):

    '''

    This methods returns the AUC Score when given the Predictions

    and Labels

    '''

    

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc
# using keras tokenizer here

token = text.Tokenizer(num_words=None)

max_len = 200



token.fit_on_texts(list(X_train) + list(X_test))

xtrain_seq = token.texts_to_sequences(X_train)

xvalid_seq = token.texts_to_sequences(X_test)



#zero pad the sequences

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)



word_index = token.word_index
%%time

with strategy.scope():

    # A simpleRNN without any pretrained embeddings and one dense layer

    model = Sequential()

    model.add(Embedding(len(word_index) + 1,

                     300,

                     input_length=max_len))

    model.add(SimpleRNN(100))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

model.summary()
model.fit(xtrain_pad, y_train,epochs=5, batch_size=64*strategy.num_replicas_in_sync) #Multipl
scores = model.predict(xvalid_pad)

print("Auc: %.2f%%" % (roc_auc(scores,y_test)))
# load the GloVe vectors in a dictionary:



embeddings_index = {}

f = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt','r',encoding='utf-8')

for line in tqdm(f):

    values = line.split(' ')

    word = values[0]

    coefs = np.asarray([float(val) for val in values[1:]])

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, 200))

for word, i in tqdm(word_index.items()):

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
with strategy.scope():

    # A simple bidirectional LSTM with glove embeddings and one dense layer

    model = Sequential()

    model.add(Embedding(len(word_index) + 1,

                     200,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

    model.add(Bidirectional(LSTM(200, dropout=0.3, recurrent_dropout=0.3)))



    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    

    

model.summary()
model.fit(xtrain_pad, y_train,epochs=5, batch_size=64*strategy.num_replicas_in_sync)
scores = model.predict(xvalid_pad)

print("Auc: %.2f%%" % (roc_auc(scores,y_test)))