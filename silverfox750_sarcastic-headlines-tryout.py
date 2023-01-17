# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import sklearn

import xgboost as xgb

import time

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_json('../input/Sarcasm_Headlines_Dataset.json',lines=True)
df.head()
#Remove Un-necessary punctuations
df['headline_new'] = df['headline'].apply(lambda x: re.sub('[^a-zA-Z]','  ',x))
# Remove the un-necessary noise from the headlines
noise_list = nltk.corpus.stopwords.words('english')
df['headline_new'] = df['headline_new'].apply(lambda x: [i for i in x.split() if i not in noise_list])
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
df['headline_new'] = df['headline_new'].apply(lambda x: [lem.lemmatize(i,'v') for i in x])
# Converting all the texts into lower case
df['headline_new'] = df['headline_new'].apply(lambda x: [i.lower() for i in x])
# Now, Let us compare both the headlines 
df[['headline','headline_new']].head(10)
df['headline_new'] = df['headline_new'].apply(lambda x: ' '.join(x))
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1400)

x = cv.fit_transform(df['headline_new']).toarray()

y = df['is_sarcastic'].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
x_train.shape
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)



#Classifier using Random Forest Classifier

#from sklearn.ensemble import RandomForestClassifier

#classifier = RandomForestClassifier(n_estimators=100)

#classifier.fit(x_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(x_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



from sklearn.metrics import accuracy_score

print("Accuracy score=",accuracy_score(y_test,y_pred)*100)
import keras

from keras.models import Sequential

from keras.layers import Dense





#Creating the L=4 layer ANN

classifier = Sequential()

classifier.add(Dense(output_dim=700,kernel_initializer = 'uniform',activation='relu',input_dim=1400))

classifier.add(Dense(output_dim=350,kernel_initializer = 'uniform',activation='relu'))

classifier.add(Dense(output_dim =175, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim =80, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim =40, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim =20, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim =10, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))





#Compile

classifier.compile(optimizer='adam',loss ='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x_train,y_train,epochs=8,batch_size=10)



y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)



from sklearn.metrics import classification_report

print(classification_report(y_pred,y_test))