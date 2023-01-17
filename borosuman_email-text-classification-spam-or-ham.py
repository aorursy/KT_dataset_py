import pandas as pd

import numpy as np

import matplotlib as plt
#Read the dataset from CSV file

df =pd.read_csv('/kaggle/input/spam.csv',encoding= 'latin-1')
#Display first 5 rows of the data frame

df.head()
#Remove unneccessary columns

df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1, inplace= True)
#Display Data Frame

df
#Rename colums

data=df.rename(columns={'v1':'class','v2':'text'})
#Display data frame

data
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
#Define features & target variable. Split train & test data

x= data['text']

y=data['class']

x_train,x_test, y_train,y_test= train_test_split(x,y,test_size=0.2)
#The text must be parsed to remove words, called tokenization. 

#Then the words need to be encoded as integers or floating point values for use as input to a machine learning algorithm,

#called feature extraction (or vectorization). So, for this purpose, CountVectorizer() is used.



v=CountVectorizer()

v.fit(x_train)

vec_x_train= v.transform(x_train).toarray()

vec_x_test= v.transform(x_test).toarray()
#Display the encoded array of the train data

vec_x_train
#display the encoded form of first row of the data

vec_x_train[0]
#display the actual tokenized word of first row of the data

v.inverse_transform(vec_x_train[0])
#display the original form of first row of the data

x_train.iloc[0]
from sklearn.naive_bayes import GaussianNB
#Here it uses Naive Bayes Classifiers for binary classification(SPAM or HAM) of text

m= GaussianNB()

m.fit(vec_x_train,y_train)

print(m.score(vec_x_test,y_test))
#Give the actual input here

sample = "Thank you for your donation We need your Complete address for Statutory purposes. Since you did not provide your complete address while donating, we are sending this communication again.  We will be grateful if you can spare 2 minutes of your valuable time to send the details.We value your continued support.  Please refer to the donation details below."

vec = v.transform([sample]).toarray()

m.predict(vec)