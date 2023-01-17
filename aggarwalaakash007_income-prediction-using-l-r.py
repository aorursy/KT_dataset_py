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
from sklearn.metrics import r2_score , accuracy_score
data = pd.read_csv('../input/adult.csv')
data = data.dropna(how = 'any')
print (data.shape)
data.loc[data['income'] == '<=50K' , 'income'] = 0
data.loc[data['income'] == '>50K' , 'income'] = 1
data = pd.get_dummies(data)
print(data)
print(data.shape)
train , test = train_test_split(data , test_size = 0.3)
output_train = train['income']
output_train.values.reshape(train.shape[0] , 1)
train = train.drop(['income'] , axis = 1)
output_validation = test['income']
output_validation.values.reshape(test.shape[0] , 1)
test = test.drop(['income'] , axis = 1)
x_train = train
y_train = output_train
y_train.values.reshape(train.shape[0] , 1)
x_validation = test
y_validation = output_validation
y_validation.values.reshape(test.shape[0] , 1)
print(x_train.shape)
print(y_train.shape)
print(x_validation.shape)
print(y_validation.shape)
model = LogisticRegression()
model.fit(x_train, y_train)
y_train_predict = model.predict(x_train)
y_predict = model.predict(x_validation)
#print y_predict
#print y_test
#print(r2_score(y_validation , y_predict))
print('accuracy of validation set: ' ,model.score(x_validation , y_validation)*100 ,'%')
#print(accuracy_score(y_validation , y_predict)*100)
print('accuracy of training set: ' ,accuracy_score(y_train , y_train_predict)*100 ,'%')

