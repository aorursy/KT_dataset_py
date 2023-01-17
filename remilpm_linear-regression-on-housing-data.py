#=========================================================================================================================

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#=========================================================================================================================



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#=========================================================================================================================

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#=========================================================================================================================

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#=========================================================================================================================

#import all the libraries

#=========================================================================================================================

from __future__ import print_function



import math



from IPython import display

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from sklearn import metrics

import tensorflow as tf

from tensorflow.python.data import Dataset

import seaborn as sns

%matplotlib inline



tf.logging.set_verbosity(tf.logging.ERROR)

pd.options.display.max_rows = 10

pd.options.display.float_format = '{:.1f}'.format




House_Price1=pd.read_csv("../input/kc_house_data.csv")



#=========================================================================================================================

#Pandas describe() is used to view some basic statistical details like  mean, std etc. of a data frame

#or a series of numeric values.

#=========================================================================================================================

House_Price1.describe()

#=========================================================================================================================

# Identify the null values and if any drop it

#=========================================================================================================================

House_Price1.isnull().sum()
#=========================================================================================================================

# Find correlation to understand how an independent variable depends on other independent variable

# if there is a 100% positive relation then value will be +1 and in case of a negative relationship it will be-1

# All the independent variables which are having positive or negative relationships should be avoided

#=========================================================================================================================

House_Price_Corr=House_Price1.corr()

House_Price_Corr

#=========================================================================================================================

#Check the correlation between price and other columns

# Correlation will be high for darker colours

#=========================================================================================================================

sns.heatmap(House_Price_Corr, xticklabels=House_Price_Corr.columns, yticklabels=House_Price_Corr['price'])
#=========================================================================================================================

#We'll randomize the data, just to be sure not to get any pathological ordering effects that might harm the 

#performance of Stochastic Gradient Descent. Additionally, we'll scale median_house_value to be in units 

#of thousands, so it can be learned a little more easily with learning rates in a range that we usually use.

#=========================================================================================================================

House_Price2 = House_Price1.reindex(np.random.permutation(House_Price1.index))

House_Price2["Median_House_Price"] = House_Price2["price"]/1000.0

House_Price2

plt.scatter(House_Price2.bedrooms,House_Price2.Median_House_Price,color="gray",label="Bed Rooms")
#=========================================================================================================================

plt.scatter(House_Price2.bathrooms,House_Price2.Median_House_Price,color="blue",label="bathrooms")

#=========================================================================================================================
plt.scatter(House_Price2.sqft_living,House_Price2.Median_House_Price,color="yellow",label="sqft_living")

plt.scatter(House_Price2.sqft_lot,House_Price2.Median_House_Price,color="green",label="sqft_lot")

plt.scatter(House_Price2.floors,House_Price2.Median_House_Price,color="red",label="floors")



plt.scatter(House_Price2.waterfront,House_Price2.Median_House_Price,color="red",label="waterfront")
plt.scatter(House_Price2.view,House_Price2.Median_House_Price,color="brown",label="view")
plt.scatter(House_Price2.condition,House_Price2.Median_House_Price,color="yellow",label="condition")
plt.scatter(House_Price2.grade,House_Price2.Median_House_Price,color="orange",label="grade")
plt.scatter(House_Price2.sqft_above,House_Price2.Median_House_Price,color="violet",label="sqft_above")
plt.scatter(House_Price2.sqft_basement,House_Price2.Median_House_Price,color="black",label="sqft_basement")
plt.scatter(House_Price2.sqft_living15,House_Price2.Median_House_Price,color="pink",label="sqft_living15")
plt.scatter(House_Price2.sqft_lot15,House_Price2.Median_House_Price,color="purple",label="sqft_lot15")
#=========================================================================================================================

# Appplying ordinary least square algorithm toreduce the squared price difference

#=========================================================================================================================

import statsmodels.formula.api as sm

House_Price_Pred_OLS=sm.ols('Median_House_Price~sqft_living+bedrooms+sqft_above+bathrooms+sqft_living15+sqft_lot15+sqft_basement',House_Price2).fit()

House_Price_Pred_OLS.params



#Conclusion

#only sqft_living, sqft_above,sqft_living15, sqft_basement and bathrooms are postively affecting house price 

House_Price_Pred_OLS.summary()
#=========================================================================================================================

# Split the data in to test and train models

#=========================================================================================================================

#lets scale the data

X = House_Price2.as_matrix(['sqft_living','bedrooms','sqft_above','bathrooms','sqft_living15','sqft_lot15','sqft_basement'])

y = House_Price2['Median_House_Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=10)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train= sc.transform(X_train)

X_test = sc.transform(X_test)

       #lets scale the data

X = House_Price2.as_matrix(['sqft_living','bedrooms','sqft_above','bathrooms','sqft_living15','sqft_lot15','sqft_basement'])

y = House_Price2['Median_House_Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=10)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train= sc.transform(X_train)

X_test = sc.transform(X_test)   
#=========================================================================================================================

#Apply linear regression

#=========================================================================================================================

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

#let us predict

y_pred=model.predict(X_test)

print (model.score(X_test, y_test))
#=========================================================================================================================

# Find the mean absolute error

#=========================================================================================================================

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

#Create XGBoost model

House_Price_Model = XGBRegressor()

#fit the model

House_Price_Model.fit(X_train, y_train)

#make predictions

predictions = House_Price_Model.predict(X_test)

# Calculate MAE

print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, y_test)))