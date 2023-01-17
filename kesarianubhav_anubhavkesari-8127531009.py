# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
import cv2

import numpy

import os

from nltk.corpus import stopwords

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import nltk

from nltk.stem.porter import *

import numpy as np

from gensim.models import word2vec

from sklearn import metrics

import pandas as pd

import time

from pandas.io.json import json_normalize

import json

from tqdm import tqdm
label_path = '../input/spotmentor/data/data/document_departments.csv'

data_path = '../input/spotmentor/data/data/docs'
labels = pd.read_csv(label_path)

df = pd.DataFrame()

for doc in tqdm(os.listdir(data_path)):

    path = os.path.join(data_path,doc)

    with open(path, 'r') as myfile:

        data=myfile.read()

    json_data = json.loads(data)

    department = labels.loc[labels['Document ID'] == int(doc[0:7])]['Department'].iloc[0]

    df = df.append(pd.DataFrame([[json_data['jd_information']['description'],department]], columns = ['Description','Department'], index = [doc[0:7]]))
print(df.shape)
print(df.columns)
from collections import Counter

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

import re

import sys

import nltk

from nltk.corpus import stopwords

import numpy as np # linear algebra

import pandas as pd 

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(df['Description']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Title")
import seaborn as sns

variety_df = df.groupby('Department').filter(lambda x: len(x) > 10)

varieties = variety_df['Department'].value_counts().index.tolist()

fig, ax = plt.subplots(figsize = (25, 10))

sns.countplot(x = variety_df['Department'], order = varieties, ax = ax)

plt.xticks(rotation = 90)

plt.show()
df['Description']==""
df = df[df.Description != '']
print(df.shape)

print(df.info())
print(df.describe())
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer(stop_words='english',use_idf=True)

model = vectorizer.fit_transform(df['Description'].str.upper())

km = KMeans(n_clusters=5,init='k-means++',max_iter=100,n_init=1)



k=km.fit(model)

terms = vectorizer.get_feature_names()

order_centroids = km.cluster_centers_.argsort()[:,::-1]

for i in range(5):

    print("cluster of words %d:" %i)

    for ind in order_centroids[i,:10]:

        print(' %s' % terms[ind])

    print() 

def decontracted(phrase):

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
v = df.apply(lambda row: decontracted(row['Description']), axis=1)

df['Description'] = v

df.head(5)
import string

punctuations=string.punctuation

def remove_punct(text):

    text  = "".join([char for char in text if char not in string.punctuation])

    text = re.sub('[0-9]+', '', text)

    return text

df['Description']=df['Description'].apply(lambda x:remove_punct(x))

df.head(10)
# nltk.download()

df['tokenized_description']=df.apply(lambda row:nltk.word_tokenize(row['Description']),axis=1 )

df.head()
ps = nltk.PorterStemmer()

def text_stemmer(text):

    text = [ps.stem(word) for word in text]

    return text
df['tokenized_description']=df['tokenized_description'].apply(lambda x: text_stemmer(x))
nltk.download('stopwords')

stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):

    text = [word for word in text if word not in stopword]

    return text

df['tokenized_description']=df['tokenized_description'].apply(lambda x: remove_stopwords(x))

df.head(3)
#generates a class map of the samples

def get_class_map(labels):

    class_map={}

    for i in labels:

        if str(i) not in class_map:

            class_map[str(i)]=1

        else:

            class_map[str(i)]+=1

#     print(class_map)

    return class_map



p=get_class_map(df['Department'].values)

print(p)

q=[i for i in p.keys()]

# p=get_class_map(labels.values)

# q=[i for i in p.keys()]

# p=get_class_map(labels.values)

sizes = [i for i in p.values()]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

plt.pie(sizes, labels=q, colors=colors,

        autopct='%1.1f%%', shadow=True)

plt.show()
class_map=p

print(p.keys())

for k in p.keys():

    if p[k]<=5:

        tempdf = df.loc[df['Department'] == k]

        df = df.append(tempdf)

        print(tempdf)
print(df.shape)
embed_size=50

input_length=100
embeddings_index = {}

f = open('../input/glove6b50dtxt/glove.6B.50d.txt')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Total %s word vectors.' % len(embeddings_index))
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()

tokenizer.fit_on_texts(df.tokenized_description)

X_data=tokenizer.texts_to_sequences(df.tokenized_description)
word_index = tokenizer.word_index

print('Total tokens.' +str( len(word_index)))
embedding_matrix = np.zeros((len(word_index) + 1, embed_size))

absent_words = 0

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

    else:

        absent_words += 1

print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)), '% of total words')
from sklearn.preprocessing import LabelEncoder

from keras.layers import Embedding

from keras.preprocessing import sequence

from keras import utils

from imblearn.over_sampling import SMOTE

from keras.layers import Input , Dense , LSTM,GlobalAveragePooling1D,GlobalMaxPooling1D,Bidirectional,LSTM,Conv1D

from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate

from keras.layers import Reshape, merge, Concatenate, Lambda, Average

from keras.models import Sequential, Model, load_model

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
le=LabelEncoder()

df['Department']=le.fit_transform(df['Department'])

df['Department'].head(2)
max_features=200000

max_senten_len=75

max_senten_num=4

embed_size=50
embedding_layer = Embedding(len(word_index)+1,

                            embed_size,

                            input_length=max_senten_len,

                            trainable=False)
X = list(sequence.pad_sequences(X_data, maxlen=max_senten_len))
print(len(X))

X=np.array(X)

print(X.shape)

Y=utils.np_utils.to_categorical(df.Department)

print(Y.shape)

print(Y[0])
from imblearn.over_sampling import SMOTE,ADASYN

#function for generating additional samples to balance the classes

def class_balancer(dataset,labels):

    sm_X,sm_Y=SMOTE(k_neighbors=1).fit_resample(dataset,labels)

#     ad_X,ad_Y=ADASYN(n_neighbors=2).fit_resample(dataset,labels)

    return sm_X,sm_Y
sm_X,sm_Y=class_balancer(X,Y)
print(sm_X.shape)

print(sm_Y.shape)
sm_X,sm_Y = shuffle(sm_X,sm_Y)

X_train,X_test,Y_train,Y_test=train_test_split(sm_X,sm_Y,test_size=0.25)
def classifier():

    inp = Input(shape=(max_senten_len,), dtype='int32')

    x = embedding_layer(inp)

    x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))(x)

    outp = Dense(27, activation="softmax")(x)

    BiLSTM = Model(inp, outp)

    BiLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    print(BiLSTM.summary())

    return BiLSTM
model12=classifier()
history=model12.fit(X_train,Y_train,batch_size=64,epochs=50,validation_data=[X_test,Y_test])
# history=model12.fit(X_train,Y_train,batch_size=64,epochs=10,validation_data=[X_test,Y_test])
# history=model12.fit(X_train,Y_train,batch_size=64,epochs=10,validation_data=[X_test,Y_test])
accs=history.history['acc']

val_accs=history.history['val_acc']

accs=accs[:10]

val_accs=val_accs[:10]

x_axis=[i+1 for i in range(10)]

plt.plot(x_axis,accs)

plt.plot(x_axis,val_accs)

plt.show()
#Creating Confusion Matrix for the Whole Dataset Based on the Trained Classififer

from sklearn.metrics import classification_report,confusion_matrix

y_trained_by_model=np.argmax(model12.predict(X_test),axis=1)

y_trained_by_model=y_trained_by_model+1

y_t=np.argmax(Y_test,axis=1)

y_t=y_t+1

print(y_trained_by_model)

print(y_t)

print(classification_report(y_t, y_trained_by_model))