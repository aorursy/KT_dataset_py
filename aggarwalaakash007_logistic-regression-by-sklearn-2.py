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
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
data = pd.read_csv('../input/Social_Network_Ads.csv')
print(data.shape)
data = data.drop(['User ID'] , axis = 1)
print(data)
data = pd.get_dummies(data)
print(data)
train , test = train_test_split(data , test_size = 0.2)
predictions = ['Age' , 'EstimatedSalary' , 'Gender_Female' , 'Gender_Male']
x_train = train[predictions]
y_train = train['Purchased']
x_validation = test[predictions]
y_validation = test['Purchased']
model = model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_validation)
#print y_predict
#print y_test
#print (r2_score(y_validation , y_predict)
print (model.score(x_validation , y_validation)) 
