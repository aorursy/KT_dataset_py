# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read in csv data to dataframe
dataset = pd.read_csv('../input/train.csv')
dataset.head()
dataset.plot.hist(y='Age',x='Survived',bins=25)
sns.boxplot(x='Age',y='Sex',data=dataset,palette='rainbow')
sns.boxplot(x="Age", y="Sex", hue="Survived",data=dataset, palette="coolwarm")
sns.lmplot(x='Fare',y='Age',data=dataset)
sns.countplot(x='Embarked',data=dataset)
plt.figure(figsize=(12,6))
sns.swarmplot(x="Pclass", y="Age",hue='Survived',data=dataset, palette="Set1", split=True)
# chop down dataset to only the features we want
train_dataset = dataset[['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']]
train_dataset.head(10)
# calc mean of age column
mean_age = int(train_dataset['Age'].mean())
# replace NaN values with mean age
train_dataset['Age'].fillna(value=mean_age,inplace=True)
train_dataset.head(10)
# split independent vars into array
X = train_dataset.iloc[:,:5].values
# split dependent var into array
y = train_dataset.iloc[:,-1].values
# load label encoder and instantiate
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
# convert gender into integer
X[:,1] = enc.fit_transform(X[:,1])
X[1]
# scale features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X[1]
# import keras tools
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
# initialise the model
model = Sequential()
# add the input layer
model.add(Dense(5,input_dim=5,kernel_initializer='normal',activation='relu'))
# add a hidden layer
model.add(Dense(6, kernel_initializer='normal',activation='relu'))
# add the output layer
model.add(Dense(1,kernel_initializer='normal'))
# add optimiser to the model and define error we want to minimise
model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
# fit the model
model.fit(X,y,epochs=100)
# load test data into dataframe
test_dataset = pd.read_csv('../input/test.csv')
test_dataset.head(2)
# create test dataframe with the features we want
t_dataset = test_dataset[['Pclass','Sex','Age','SibSp','Parch','Fare']]
test_mean_age = int(t_dataset['Age'].mean())
test_mean_age
# replace NaN values with mean age
t_dataset['Age'].fillna(value=test_mean_age,inplace=True)
t_dataset.tail()
# convert features into array
X_test = t_dataset.iloc[:,:5].values
X_test[1]
# encode gender cals
X_test[:,1] = enc.fit_transform(X_test[:,1])
# scale values
X_test = sc.fit_transform(X_test)
# save out predictions
X_pred = model.predict(X_test)
X_pred[1]
# convert to bool > 5 means prediction is survived
X_pred_bool = (X_pred > 0.5)
len(X_pred_bool)
test_dataset['Survived'] = X_pred_bool
test_dataset.tail(50)
