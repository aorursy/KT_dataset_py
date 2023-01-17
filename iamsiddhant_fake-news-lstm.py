import pandas as pd
data=pd.read_csv('../input/fake-news/train.csv')

data.head()



#we will be using the title column for our prediction
#checking for null values in the dataset



data.isnull().sum()
data.shape
#we will use the title column so other columns will be of no use



data=data.drop(['text','author','id'],axis=1)
#there are some  null values in the title column also



data.isnull().sum()
#as title is the only column is the what we are using if it contains NaN values we have to drop it.



data=data.dropna()
data.isnull().sum()
data.shape
data.head()
X=data['title']

y=data['label']
X.shape
#importing all necessary modules that we will be using to build our LSTM neural network



import tensorflow as tf

from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense
#we dropped some rows as there were nan values so reset index will make it uniform



X=X.reset_index()
X=X.drop(['index'],axis=1)
X.tail()
#as we dropped some rows so to make the dataframe in order

y=y.reset_index()
y=y.drop(['index'],axis=1)
y.tail()
# importing nltk,stopwords and porterstemmer we are using stemming on the text we have and stopwords will help in removing the stopwords in the text



#re is regular expressions used for identifying only words in the text and ignoring anything else

import nltk

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

corpus = []

#each row of the dataset is considered here.everything except the alphabets are removed ,stopwords are also being removed here .the text is converted in lowercase letters and stemming is performed

#lemmatisation can also be used here at the end a corpus of sentences is created

for i in range(0, len(X)):

    review = re.sub('[^a-zA-Z]', ' ',X['title'][i])

    review = review.lower()

    review = review.split()

    

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)
corpus[30]
#vocabulary size

voc_size=5000
#performing onr hot representation



onehot_repr=[one_hot(words,voc_size)for words in corpus] 
len(onehot_repr[0])
len(onehot_repr[700])
#specifying a sentence length so that every sentence in the corpus will be of same length



sent_length=25



#using padding for creating equal length sentences





embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

print(embedded_docs)
#Creating model



from tensorflow.keras.layers import Dropout

embedding_vector_features=40

model=Sequential()

model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))

model.add(Dropout(0.3))

model.add(LSTM(200))

model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

import numpy as np

X_final=np.array(embedded_docs)

y_final=np.array(y)
X_final.shape,y_final.shape
#splitting the data for training and testing the model



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.10, random_state=42)
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64)
#loading test dataset for prediction



test=pd.read_csv('../input/fake-news/test.csv')
test.head()
#null values in the test dataset



test.isnull().sum()
#using the title column only as we did in the train dataset



test=test.drop(['text','id','author'],axis=1)
test.head()
test.isnull().sum()
test.fillna('fake fake fake',inplace=True)



#the solution file that can be submitted in kaggle expects it to have 5200 rows so we can't drop rows in the test dataset
test.shape
#creating corpus for the test dataset exactly the same as we created for the training dataset



corpus_test = []

for i in range(0, len(test)):

    review = re.sub('[^a-zA-Z]', ' ',test['title'][i])

    review = review.lower()

    review = review.split()

    

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus_test.append(review)
#creating one hot representation for the test corpus



onehot_repr_test=[one_hot(words,voc_size)for words in corpus_test] 
#padding for the test dataset

sent_length=25



embedded_docs_test=pad_sequences(onehot_repr_test,padding='pre',maxlen=sent_length)

print(embedded_docs_test)
X_test=np.array(embedded_docs_test)
#making predictions for the test dataset



check=model.predict_classes(X_test)
check
check.shape
test.shape
submit_sample=pd.read_csv('../input/fake-news/submit.csv')
submit_sample.head()
type(check)
check[0]
val=[]

for i in check:

    val.append(i[0])
#inserting our predicted values in the submission file



submit_sample['label']=val
submit_sample.head()
#saving the submission file



submit_sample.to_csv('submission.csv',index=False)