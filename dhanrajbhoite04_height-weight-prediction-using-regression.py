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
df=pd.read_csv('../input/weight-height.csv')
df
# pre-process on the data fro dependent and independet variable find out

# X is the independent variable it contain Gender and Height

X = df.iloc[:, :-1].values

# y is the depedent varibale it contain dependent variable i.e Weight

y = df.iloc[:, 2].values
print(X)

#it gives array with gender and height which is independent variable
# it gives the wight array which dependent variable

print(y)
# So look at the X array which have data gender which is text format but for machine learning

#need to convert it into the number so we nned to do data processing on that which have using skleam

#preprocessing the datata ex. male female are lable converted into categorial data like 0 and 1

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

labelencoder_X = LabelEncoder()

X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])

X = onehotencoder.fit_transform(X).toarray()



#it will make if the male it it gives array [1,0,71.12123]
print(X)

# see the array of matrix which is contian 1 and 0 lable for male and female
# here we split the data into trin and test data

#here we take test data 0.2 means 20% of data take into test data remain 80% will train data

#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.2, random_state = 0)
# herer we actual take regression model and tain our data and fit the model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
# this gives the actual prediction from our train data we give the input X_test we alrady splited 

y_pred = regressor.predict(X_test)
#here is the actual  prediction of wight

print(y_pred)
#see the y_pred is predict from our model and y_test which is actual data which very close to our prediction

print(y_test)
#Example

# if 1 is in first column then he is male and 1 is in second column the she is female 

y_pred = regressor.predict([[1., 0., 73.847017017515],[0., 1., 67.46708591]])
print(y_pred)