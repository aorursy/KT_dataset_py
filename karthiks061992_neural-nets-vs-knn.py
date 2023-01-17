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
dataset=pd.read_csv('/kaggle/input/epl-stats-20192020/epl2020.csv')

dataset.columns
features=['h_a','xG','xGA','npxG','npxGA','deep','deep_allowed','scored','missed','xpts','result']

dataset=dataset[features]

dataset.head()
X=dataset.iloc[:,:-1]

Y=dataset.iloc[:,10]

Y.head()

from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()

X.iloc[:,0]=enc.fit_transform(X.iloc[:,0])

X.head()

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

Y=pd.Series(Y)

Y=Y.values.reshape(-1,1)

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")

Y=ct.fit_transform(Y)
print(Y)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)

import keras

from keras.models import Sequential

from keras.layers import Dense
classifier=Sequential()

classifier.add(Dense(input_dim=10,output_dim=5,kernel_initializer="uniform",activation="relu"))

classifier.add(Dense(output_dim=5,kernel_initializer="uniform",activation="softmax"))

classifier.add(Dense(output_dim=5,kernel_initializer="uniform",activation="softmax"))

classifier.add(Dense(output_dim=3,kernel_initializer="uniform",activation="softmax"))

classifier.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])

classifier.fit(xtrain,ytrain,batch_size=30,epochs=100)

ypred=classifier.predict(xtest)

#I would request comments on this is the neural network  overfitting at one point of time as of now i have seen an accuracy

#of 74 percent sometimes i do see the accuracy popping up suddenly to 89 or so
#classification using KNN

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(xtrain,ytrain)

knn.score(xtest,ytest)

#fetching an accuracy of 68 percent with KNN
#with neigbouring point of just 1

#then

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(xtrain,ytrain)

knn.score(xtest,ytest)