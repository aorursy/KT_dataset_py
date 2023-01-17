# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/churn-modelling-dataset/Churn_Modelling.csv')
df.head()
df.info()
df.describe(include='all')
df.shape
x=df.iloc[:,3:13].values
print(x[0:5])
y=df.iloc[:,13].values
print(y[0:5])
# Encoding Categorical data

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

labelencoder_X_1=LabelEncoder()

x[:,1]=labelencoder_X_1.fit_transform(x[:,1])

labelencoder_X_2=LabelEncoder()

x[:,2]=labelencoder_X_1.fit_transform(x[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])



x=onehotencoder.fit_transform(x).toarray()
print(x[0])
x=x[:,1:]
# Splitting datasets into training and testing test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=365)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)

x_test
# Part 2 Building the ANN

# Importing Keras Library

import keras

from keras.models import Sequential

from keras.layers import Dense
# Intializing the ANN

classifier=Sequential()
# Adding the input and the first hidden layer

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
# Adding the second hidden layer

classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
# Adding the output layer

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
# Compiling the ANN

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# Fitting the dataset into ANN

classifier.fit(x_train,y_train,batch_size=10,epochs=100)
# Part 3 Prediction of the test data

y_hat=classifier.predict(x_test)
y_hat=(y_hat>0.5)
from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_hat)

print(cm)

ac=accuracy_score(y_test,y_hat)

print(ac)