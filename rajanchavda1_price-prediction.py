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
dataset = pd.read_csv('../input/car data.csv')
dataset.head(5)
X = dataset.drop(['Present_Price', 'Car_Name'] ,axis=1 )

#Fuel_Type --> 1 = Petrol , 0 = Diesel
#Seller_Type --> 1 = Manual , 0 = Automatic 
#Seller_Type --> 1 = Dealer , 0 = Individual

X['Fuel_Type'] = X.Fuel_Type.apply(lambda x: 1 if x == 'Petrol' else 0)
X['Seller_Type'] = X.Seller_Type.apply(lambda x: 1 if x == 'Dealer' else 0)
X['Transmission'] = X.Transmission.apply(lambda x: 1 if x == 'Manual' else 0)

y = dataset.Present_Price
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=56)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import svm
classifiers = [
    svm.SVR(),
    linear_model.BayesianRidge(),
    linear_model.ARDRegression(),
    linear_model.LinearRegression()]
for item in classifiers:
    print(item)
    clf = item
    clf.fit(X_train, y_train)
    print('\nScore: ',clf.score(X_train, y_train))
    y_pred = clf.predict(X_test) 
#     print(y_pred)
    print('Mean Squared Error: ', mean_squared_error(y_test, y_pred),'\n\n\n')

