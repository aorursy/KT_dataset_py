%matplotlib inline

import warnings

warnings.filterwarnings("ignore")





import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os
# using SQLite Table to read data.

con = sqlite3.connect('../input/database.sqlite') 



# filtering only positive and negative reviews i.e. 

# not taking into consideration those reviews with Score=3

# SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000, will give top 500000 data points

# you can change the number to any other number based on your computing power

filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 50000""", con) 

# for tsne assignment you can take 5k data points



#filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 100000""", con) 

#filtered_data=pd.read_csv('Reviews.csv')



# Give reviews with Score>3 a positive rating(1), and reviews with a score<3 a negative rating(0).

def partition(x):

    if x < 3:

        return 0

    return 1



#changing reviews with score less than 3 to be positive and vice-versa

actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition) 

filtered_data['Score'] = positiveNegative

print("Number of data points in our data", filtered_data.shape)

filtered_data.head(3)
display = pd.read_sql_query("""

SELECT UserId, ProductId, ProfileName, Time, Score, Text, COUNT(*)

FROM Reviews

GROUP BY UserId

HAVING COUNT(*)>1

""", con)
print(display.shape)

display.head()
display[display['UserId']=='AZY10LLTJ71NX']
display['COUNT(*)'].sum()
display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND UserId="AR5J8UI46CURR"

ORDER BY ProductID

""", con)

display.head()
#Sorting data according to ProductId in ascending order

sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Deduplication of entries

final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

final.shape
#Checking to see how much % of data still remains

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND Id=44737 OR Id=64422

ORDER BY ProductID

""", con)



display.head()
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
#Before starting the next phase of preprocessing lets see the number of entries left

print(final.shape)



#How many positive and negative reviews are present in our dataset?

final['Score'].value_counts()
# printing some random reviews

sent_0 = final['Text'].values[0]

print(sent_0)

print("="*50)



sent_1000 = final['Text'].values[1000]

print(sent_1000)

print("="*50)



sent_1500 = final['Text'].values[1500]

print(sent_1500)

print("="*50)



sent_4900 = final['Text'].values[4900]

print(sent_4900)

print("="*50)
# remove urls from text python: https://stackoverflow.com/a/40823105/4084039

sent_0 = re.sub(r"http\S+", "", sent_0)

sent_1000 = re.sub(r"http\S+", "", sent_1000)

sent_150 = re.sub(r"http\S+", "", sent_1500)

sent_4900 = re.sub(r"http\S+", "", sent_4900)



print(sent_0)
# https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element

from bs4 import BeautifulSoup



soup = BeautifulSoup(sent_0, 'lxml')

text = soup.get_text()

print(text)

print("="*50)



soup = BeautifulSoup(sent_1000, 'lxml')

text = soup.get_text()

print(text)

print("="*50)



soup = BeautifulSoup(sent_1500, 'lxml')

text = soup.get_text()

print(text)

print("="*50)



soup = BeautifulSoup(sent_4900, 'lxml')

text = soup.get_text()

print(text)
# https://stackoverflow.com/a/47091490/4084039

import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
sent_1500 = decontracted(sent_1500)

print(sent_1500)

print("="*50)
#remove words with numbers python: https://stackoverflow.com/a/18082370/4084039

sent_0 = re.sub("\S*\d\S*", "", sent_0).strip()

print(sent_0)
#remove spacial character: https://stackoverflow.com/a/5843547/4084039

sent_1500 = re.sub('[^A-Za-z0-9]+', ' ', sent_1500)

print(sent_1500)
# https://gist.github.com/sebleier/554280

# we are removing the words from the stop words list: 'no', 'nor', 'not'

# <br /><br /> ==> after the above steps, we are getting "br br"

# we are including them into stop words list

# instead of <br /> if we have <br/> these tags would have revmoved in the 1st step



stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"])
# Combining all the above stundents 

from tqdm import tqdm

preprocessed_reviews = []

# tqdm is for printing the status bar

for sentance in tqdm(final['Text'].values):

    sentance = re.sub(r"http\S+", "", sentance)

    sentance = BeautifulSoup(sentance, 'lxml').get_text()

    sentance = decontracted(sentance)

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    # https://gist.github.com/sebleier/554280

    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)

    preprocessed_reviews.append(sentance.strip())
import tensorflow as tf

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

    raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

# fix random seed for reproducibility

np.random.seed(7)


X=preprocessed_reviews

y=np.array(final['Score'])
from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

print(len(X_train))

print(len(X_test))
#https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do

from keras.preprocessing.text import Tokenizer

token=Tokenizer(num_words=5000)

token.fit_on_texts(X_train)

X_train= token.texts_to_sequences(X_train)

X_test=token.texts_to_sequences(X_test)

print(X_train[0])

max_review_length = 600

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)



print(X_train.shape)

print(X_train[1],y_train[1])
import matplotlib.pyplot as plt

import numpy as np



def plt_dynamic(x, vy, ty, ax, colors=['b']):

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()

    fig.canvas.draw()
# create the model

epoch=10

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(5000, embedding_vecor_length, input_length=max_review_length))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
from keras.optimizers import Adam

batch_size=32

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

history=model.fit(X_train, y_train,batch_size=batch_size,epochs=epoch,verbose=1,validation_data=(X_test, y_test))




score = model.evaluate(X_test, y_test, verbose=0) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_title('Epoch vs Loss')

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epoch+1))





vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)



# create the model

from keras.layers import Dropout

epoch=10

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(5000, embedding_vecor_length, input_length=max_review_length))

model.add(LSTM(100,return_sequences=True))

model.add(Dropout(0.5))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
from keras.optimizers import Adam

batch_size=32

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

history=model.fit(X_train, y_train,batch_size=batch_size,epochs=epoch,verbose=1,validation_data=(X_test, y_test))




score = model.evaluate(X_test, y_test, verbose=0) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_title('Epoch vs Loss')

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epoch+1))





vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)


