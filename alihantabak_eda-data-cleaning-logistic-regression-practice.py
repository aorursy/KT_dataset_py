# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Firstly, we should import data from dataset
data = pd.read_csv("../input/seattleWeather_1948-2017.csv")
data.head()
#Shows us informations of first 5 weather conditions.
data.tail()
#Shows us informations of last 5 weather conditions.
data.columns #See columns in data
data.describe()
data.drop(["DATE"],axis=1,inplace=True) #Drop processing
data.columns #As you can see, we droped DATE feature
data.info()
data=data.dropna() #We drop all NaN values.
data.info()
data.RAIN = [1 if each == True  else 0 for each in data.RAIN.values]
data.info()
#You can see easily; RAIN values converted to integer.
y = data.RAIN.values
#our y axis is defined RAIN values because we want to learn how weather is? 
#So we are trying to get: the weather is rainy or not.
x_data = data.drop(["RAIN"],axis=1)
#x_data is all of features except RAIN in the data.
x_data
#Normalization
x = (x_data - np.min(x_data))/(np.max(x_data)).values
x
#Train-Test Datas Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
#test_size=0.2 means %20 test datas, %80 train datas
x_train.head()
x_train.tail()
y_test
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T
from sklearn import linear_model
logistic_reg = linear_model.LogisticRegression(random_state=42,max_iter=150)
#max_iter is optional parameter. You can write 10 or 3000 if you want.
print("Test accuracy {}".format(logistic_reg.fit(x_train.T,y_train.T).score(x_test.T,y_test.T)))
print("Train accuracy {}".format(logistic_reg.fit(x_train.T,y_train.T).score(x_train.T,y_train.T)))