# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pprint import pprint

import itertools

import nltk

import string

import re

from sklearn.model_selection import train_test_split

import sklearn

from numpy import array

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten

from keras.layers import GlobalMaxPooling1D

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from tensorflow.keras.layers import LSTM

from keras.layers import Conv1D

from numpy import array

from numpy import asarray

from numpy import zeros

import re



from wordcloud import WordCloud 

import matplotlib.pyplot as plt 
df = pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
df.shape
df['salary_range'].head()
df1 =df.copy()
salary_range = df1['salary_range'].copy()
salary_range.fillna('0-0',inplace=True)
salary_range.replace('0','0-0',inplace=True)
for i in range(len(salary_range)):

    sal = re.findall('\d[0-9]*',salary_range[i])

    if len(sal)==2:

        mean = (int(sal[0])+int(sal[1]))//2

    else:

        mean = int(sal[0])

    salary_range[i] = mean

# we will divide the salary range into 5 groups and plot its histogram 

temp_sal_range = salary_range.copy()

temp_sal_range = temp_sal_range[temp_sal_range<100000000]

temp_sal_range.max()
def create_bins(df):

    bin1=0

    bin2=0

    bin3=0

    bin4=0

    bin5=0

    for i in range(len(df)):

        if df[i]==0:

            bin1+=1

        elif 0<df[i]<=40000:

            bin2+=1

        elif 40001<=df[i]<=100000:

            bin3+=1

        elif 100001<=df[i]<=250000:

            bin4+=1

        elif df[i]>250000:

            bin5+=1

    return [bin1,bin2,bin3,bin4,bin5]



sal_list = create_bins(salary_range)

sal_list

            
# Plotting the histogram of various salaries provided 

plt.bar([1,2,3,4,5],sal_list)

plt.xticks([1,2,3,4,5],['0 or not mentioned','40000<','100000<','250000<','250000>'],

          rotation=45)

plt.xlabel('Sal_range')

plt.ylabel('Applications')

plt.plot()
loc = df['location'].copy()

len(loc)
loc_list =[]

loc.replace(np.NaN,'0',inplace=True)

for i in range(len(loc)):

    if loc[i]!=np.nan:

        country = re.findall('\w[A-Z]*',loc[i])

        loc_list.append(country[0])

loc_list = pd.Series(loc_list)

# there are lot of countries with 1 or 2 applications so we'll plot the ones 

# with max applications 

plt.bar([x for x in range(11)],loc_list.value_counts()[1:12])

plt.xlabel("Country")

plt.ylabel("Applications")

plt.xticks([x for x in range(11)],loc_list.value_counts().index[1:12])

plt.plot()
# as most no of postings were from US it was acting  as an outlier 

# anyways I will plot the graph too 

# there are lot of countries with 1 or 2 applications so we'll plot the ones 

# with max applications 

plt.bar([x for x in range(11)],loc_list.value_counts()[:11])

plt.xlabel("Country")

plt.ylabel("Applications")

plt.xticks([x for x in range(11)],loc_list.value_counts().index[:11])

plt.plot()
# top 20 Department options 

plt.figure(figsize=(20,10))

plt.bar([x for x in range(21)],df['department'].value_counts()[:21])

plt.xticks([x for x in range(21)],df['department'].value_counts().index[:21],rotation=90)

plt.xlabel('Department')

plt.ylabel('No of applications')

plt.plot()
#plotting different employment types using pie chart 

labels = ['Full-time','Contract','Part-time','Temporary','Other']

sizes =[df['employment_type'].value_counts()[x] for x in range(5)]

explode = (0,0.1,0.2,0.3,0.4)



#plot

plt.pie(sizes,labels=labels,explode=explode,

       shadow=True,startangle=45)

plt.axis('equal')

plt.show()
x =df1[df1['fraudulent']==1]



# word cloud of the job description 

words = x['description'][~pd.isnull(x['description'])]

wordcloud = WordCloud(width=500,height=400).generate(''.join(words))



plt.figure(figsize=(10,12))

plt.axis('off')

plt.title('Likely to be fraudulent...')

plt.imshow(wordcloud)

plt.show()
df.fillna(' ',inplace=True)
#concatenating  all the columns with text

df['features']=df['title']+" " + df['department'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits'] + " " 
''' Removing spaces ,punctuations,numbers,urls'''

def regex(text):

    text = text.replace("  "," ")

    text= text.lower()

    text =re.sub(r"http\S+", "", text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('\n', '', text)

    text = re.sub(r'[^\w\s]', '', text) 

    text = re.sub('\w*\d\w*', '', text)

    return text



df['features']=df['features'].apply(lambda x: regex(x))
df1 = df[['features','fraudulent']].copy()
#Tokenizer

maxlen=100

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(df1['features'])



df1['features'] = tokenizer.texts_to_sequences(df1['features'])
# Padding

vocab_size = len(tokenizer.word_index) + 1



df1['features']= pad_sequences(df1['features'], padding='post', maxlen=100)
# using glove6b.txt for creating an embedding dictionary.

embeddings_dictionary = dict()

with open('../input/glove6b/glove.6B.100d.txt', encoding="utf8") as glove_file:

    for line in glove_file:

        records = line.split()

        word = records[0]

        vector_dimensions = asarray(records[1:], dtype='float32')

        embeddings_dictionary [word] = vector_dimensions

glove_file.close()
embedding_matrix = zeros((vocab_size, 100))

for word, index in tokenizer.word_index.items():

    embedding_vector = embeddings_dictionary.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector
# splitting the dataset into training and testing.

x_train,x_test,y_train,y_test = train_test_split(df1['features'],df1['fraudulent']

                                                ,test_size=0.25)
def Lstm():

    model = Sequential()

    embedding_layer = Embedding(vocab_size,100,weights=[embedding_matrix],

                               input_length=maxlen,trainable=False)

    model.add(embedding_layer)

    model.add(LSTM(128))



    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    

    return model
#overview of model

model = Lstm()

model.summary()
history = model.fit(x_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)



score = model.evaluate(x_test, y_test, verbose=1)
print("Test Accuracy:", score[1])
#lstm 

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','test'], loc = 'upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'], loc = 'upper left')

plt.show()
#convolutional neural networks 

def CNN():

    model = Sequential()



    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)

    model.add(embedding_layer)



    model.add(Conv1D(128, 5, activation='relu'))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    

    return model 
model2 = CNN()

model2.summary()
history2 = model.fit(x_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)



score2 = model.evaluate(x_test, y_test, verbose=1)
print("Test Accuracy:", score2[1])
#cnn 

plt.plot(history2.history['acc'])

plt.plot(history2.history['val_acc'])



plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper left')

plt.show()



plt.plot(history2.history['loss'])

plt.plot(history2.history['val_loss'])



plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper left')

plt.show()
#references 

# https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/