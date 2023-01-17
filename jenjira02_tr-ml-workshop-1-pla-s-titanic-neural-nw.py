# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#1. Read data

df = pd.read_csv('../input/titanic_data.csv')

df.shape
df.head()
#2. Clean data

#2.1 Drop unused data

df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1) 



df = df.dropna() #delete data with NA information

df.head() #Plot table again
#2.2 Change Category values e.g. Sex into numbers.

sexConvDict = {"male":1 ,"female" :2}

df['Sex'] = df['Sex'].apply(sexConvDict.get).astype(int)

df.head()
#3. [Prediction] Define columns that will be inputs used for prediction (called Features)

#3.1 Define fields for prediction

features = ['Sex','Parch','Pclass', 'Age', 'Fare', 'SibSp']

df[features]
#3.2 Use Standard Scaler to arrange data to have the same range & distribution

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



#X = input used to transform and create predictions, y = prediction

#First way use scaler to arrange data in the same range

X = scaler.fit_transform(df[features].values)

#Second way not use scaler -> worse result

#X = df[features].values



y = df['Survived'].values
X
# 3.3 Get scalar input X to train and use the training result to test

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train.shape
#3.4 Create neural network model using train data

from sklearn.neural_network import MLPClassifier as mlp



#We can change neural network algorithm here

nn = mlp(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(3, 2), random_state=0)

nn.fit(X_train, y_train) 
#3.5 Use the trained model to predict test data

predicted = nn.predict(X_test)
#3.6 Measure the accuracy of data

from sklearn.metrics import accuracy_score



accuracy_score(y_test, predicted)
#4. Draw the table of prediction results to see which ones are predicted correctly and which ones not.

from sklearn.metrics import confusion_matrix



confusion_matrix(y_test, predicted)



#How to read this confusion matrics

#We want numbers in top left and bottom right to be the most = better accuracy

# Actual\Predict No   Yes

# No            102

# Yes                 50