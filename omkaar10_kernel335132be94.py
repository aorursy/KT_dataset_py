# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as mp

from sklearn.linear_model import LinearRegression #for the learning model

from sklearn.model_selection import train_test_split#to split the data into 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
housing=pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
housing.head()
housing.columns
#dropping ocean_proximity column

housing=housing.drop('ocean_proximity',axis=1)
print('Shape :',housing.shape)

print('Number of nulls :',housing.isna().sum())

print('Datatypes :', housing.dtypes)
#we have to remove the null values

#best way to do so is to either fill them with mean or delete them.



housing=housing.dropna(axis=0)

housing.shape
#We will divide the features into dependent and independent features



X=housing.drop('median_house_value',axis=1)

Y=housing['median_house_value'].values



print('Shape of X :',X.shape)

print('Shape of Y :',Y.shape)
regressionObject=LinearRegression() 
#We will now split our data into training an testing sets viz, X_train,Y_train,X_test,Y_test



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2) #here test_size=0.2 means 20% of original data is taken for testing and 80% as trainning



#you can check the by firing the shape command for test and train sets

print(X_train.shape)

print(X_test.shape)
#fitting the sets



regressionObject.fit(X_train,Y_train)
# now we will start the prediction for y



y_pred=regressionObject.predict(X_test)
#Lets check the prediction value

print('Predicted Value :',y_pred[0])

print('Actual Value :', Y_test[0])
#now we will create a dataframe of predicted values and actual values



res=pd.DataFrame({'Actual house value':Y_test,'Predicted house value':y_pred})

res=res.reset_index()

print(res.columns)

res=res.drop(['index'],axis=1) #dropping the index column

print(res.columns)
#Plotting a graph for comparision between our predicted values and the actual values



mp.plot(res[:50])

mp.legend(['Actual','Predicted'])

mp.show()
res[0:50]
#Checking the score



print('Train data score :',regressionObject.score(X_train,Y_train))

print('Test data score :',regressionObject.score(X_test,Y_test))