# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#loading the dataset
train_dataset=pd.read_csv('../input/train.csv')
#Describing the dataset
train_dataset.describe()
#Splitting the Independent and Dependent Variable
x_train=train_dataset.iloc[:,[2,4]].values
y_train=train_dataset.iloc[:,1].values
#labelEncoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x_train[:,1]=labelencoder.fit_transform(x_train[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
x_train=onehotencoder.fit_transform(x_train).toarray()
#Building the model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)      
classifier.fit(x_train,y_train)
#prediction
t_dataset=pd.read_csv('../input/test.csv')
x_test=t_dataset.iloc[:,[1,3]].values
sc_xtest=LabelEncoder()
x_test[:,1]=labelencoder.fit_transform(x_test[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
x_test=onehotencoder.fit_transform(x_test).toarray()
y_pred=classifier.predict(x_test)