# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score

from scipy import stats

import seaborn as sns



#Read data

data = pd.read_csv("../input/Admission_Predict.csv", header=0) 





y=data["Chance of Admit "]                                                 #Setting the response variable y 

data= data.drop("Serial No.",axis=1)                                       #Dropping Column serial number 

data["Research"]= data["Research"].astype("category")                      #Changing the type to categorical

data["University Rating"]= data["University Rating"].astype("category")    #Changing the type to categorical

data= data.drop("Chance of Admit ",axis=1)                                 #Dropping Column Chance of Admit from X



#Doing the BoxCox Transformation

bc,lamb=stats.boxcox(y)

y=(pow(y,lamb)-1)/lamb          #Applying the BoxCox transformation formula using the lamba obtained in the previous step
#Performing the Recursive Feature elimination

model=LinearRegression()

rfe = RFE(model, 3)

fit = rfe.fit(data, y)

for i in range(len(data.columns)):

    if fit.ranking_ [i]!=1:

        data=data.drop(data.columns[i],axis=1)

        

#Splitting the data into train and test set

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2,random_state=30)



#Applying Linear Regression

clf = LinearRegression().fit(X_train, y_train)

pred=clf.predict(X_test)          #Predicting the Chance of Admit



#Calculating the mean squared error and R-squared value

print("MSE:",mse(y_test,pred))

print("R2:", r2_score(y_test,pred))



#Plotting the residual graph to check the linear regression assumptions

sns.residplot((y_test-pred), pred, lowess=True, color="g")