# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # For Visualisation of data 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_boston

mydata = load_boston()
print(mydata.keys())
mydf=pd.DataFrame(mydata.data, columns=mydata.feature_names)

mydf.head()
mydf["Target"]=mydata.target
import seaborn as sns

correlation=mydf.corr()

plt.figure(figsize=(20, 5))

sns.heatmap(data=correlation, annot=True)

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']

target = mydf['Target']



for i, col in enumerate(features):

    plt.subplot(1, len(features) , i+1)

    x = mydf[col]

    y = target

    plt.scatter(x, y, marker='o')

    plt.title(col)

    plt.xlabel(col)

    plt.ylabel('Target')
X=pd.DataFrame(np.c_[mydf['LSTAT'], mydf["RM"]], columns = ['LSTAT', 'RM'])

Y=mydf['Target']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 5)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
Y_predict_Train=model.predict(X_train)

rmse = (np.sqrt(mean_squared_error(Y_train, Y_predict_Train)))

r2 = r2_score(Y_train, Y_predict_Train)



print("The model performance for training set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))

print("\n")
Y_predict_Test=model.predict(X_test)

rmse = (np.sqrt(mean_squared_error(Y_test, Y_predict_Test)))

r2 = r2_score(Y_test, Y_predict_Test)



print("The model performance for training set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))

print("\n")