# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model as lm

from sklearn.feature_selection import RFE #recursive feature elimination

import matplotlib.pyplot as plt

# function to calculate r-squared, MAE, RMSE

from sklearn.metrics import r2_score , mean_absolute_error,mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/carPrice.csv')
df.head()
df.columns.values
#describe data

df.describe()
##create dummy variables for the column

dummy_cols = """carCompany fueltype aspiration doornumber carbody 

drivewheel enginelocation enginetype 

cylindernumber fuelsystem""".split()



dummies = pd.get_dummies(df[dummy_cols])

dummies.columns.values
#drop the original column

df = df.drop(dummy_cols, axis=1)
df.columns.values
##add dummy variabale

df = df.join(dummies)
df.columns.values
#Break data up into training and test datasets

#creat out predictor/independent variable

#and our response/dependent variable

x = df.drop(['price','car_ID'],axis = 1)

y = df['price']

col = x.columns.values
# Normalizing the data

from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(x)

X_std = std_scale.transform(x)



X = pd.DataFrame(X_std,columns = col)
X.columns.values
X.head()
## correlation Matrix

corr = X.corr()

print(corr)
#create test data from the first 150 observations

X_train = X[0:150]

y_train = y[0:150]



## create training data form the remaining observations

X_test = X[150:]#create an object that is an ols regression

y_test = y[150:]
X_test.columns.values
#Train the linear Model

#create an object that is an ols regression

ols = lm.LinearRegression()
#Train the model using our training data

model = ols.fit(X_train, y_train)
model.intercept_
#view the training model's coefficient

model.coef_
#Run the model on X_test and show the first five results

list(model.predict(X_test)[0:5])
# View the R-Squared score



Price_Pred = model.predict(X_test)

r_squared = r2_score(Price_Pred, y_test)

r_squared
# Adjusted R Squared

1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
# View the first five test Y values

list(y_test)[0:5]
# Apply the model we created using the training data 

# to the test data, and calculate the RSS.

((y_test - model.predict(X_test)) **2).sum()


# Calculate the MSE

np.mean((model.predict(X_test) - y_test) **2)