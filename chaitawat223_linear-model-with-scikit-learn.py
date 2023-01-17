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
df = pd.read_csv("../input/insurance.csv")
df.head(10)
#Convert string to int by "get_dummies"

sex_cat = pd.get_dummies(df['sex'],drop_first=True,prefix='sex')
smoker_cat = pd.get_dummies(df['smoker'],drop_first=True,prefix='smoker')
region_cat = pd.get_dummies(df['region'],drop_first=True,prefix='region')

dummies = pd.concat([sex_cat,smoker_cat,region_cat],axis=1)
print(dummies)
#merge to the original table

df_withdummies = pd.concat([df,dummies],axis=1)
df_withdummies.head(10)
#Linear regression with scikit-learn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Split dataset
#Drop columns containing string value
#Replace them with "get_dummies"
x = df_withdummies.drop(columns=['charges','sex','smoker','region'])
y = np.array(df_withdummies['charges'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

print(x.shape)
print(y.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
linear = LinearRegression()

#fit linear regression
linear.fit(x_train,y_train)

#Caluculateing error in...
#Training data
prediction_train = linear.predict(x_train)

prediction_train_error = []
for i in prediction_train:
    train_error = (y_train - prediction_train)
    prediction_train_error.append(train_error)
    train_MSE = np.mean(np.abs(train_error))/len(prediction_train_error)

#Testing data
prediction_test = linear.predict(x_test)

prediction_test_error = []
for j in prediction_test:
    test_error = (y_test - prediction_test)
    prediction_test_error.append(test_error)
    test_MSE = np.mean(np.abs(test_error))/len(prediction_test_error)

print('mean absolute error of training set: {}'.format(train_MSE))
print('mean absolute error of testing set: {}'.format(test_MSE))
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))

#Ploting prediction error in training dataset
plt.subplot(2,1,1)
plt.plot(np.array(train_error))

#Ploting prediction error in test dataset
plt.subplot(2,1,2)
plt.plot(np.array(test_error))
#gets coefficients
linear.coef_
#original data
x
