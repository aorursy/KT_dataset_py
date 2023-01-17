# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
#display(data.head())
#data = data.dropna()
datacopy = data.copy()
#print(datacopy.head())
y= datacopy['diagnosis']
y = pd.get_dummies(y)
X=datacopy.drop(columns =['diagnosis','Unnamed: 32','id'],axis =1)
#my_imputer = SimpleImputer()
#X_impute = my_imputer.fit_transform(X)
#print(X.head())
(train_x,test_x,train_y,test_y) = train_test_split(X,y)
#print(test_y)
linear_model = LinearRegression()
linear_model.fit(train_x,train_y)
predict_result = linear_model.predict(test_x)
#print(predict_result)
print("error rate" ,mean_absolute_error(test_y, predict_result)*100)
linear_model.score
result_precntage = round(linear_model.score(test_x, test_y) * 100, 2)
print("Success rate" ,result_precntage)
