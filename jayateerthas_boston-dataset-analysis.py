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
#importing the necessary libraries such as numpy, pandas, matplotlib and seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#the required data set is available in sklearn only. Hence the boston housing data is available from sklearn
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
# the below shows the details under the dataset. 'data' is the actual data. feature_names are the name of the columns of data.
#target is the dependant variable which is the price of the houses. DESCR gives the details of the dataset. We can access all 
#details under the keys using 'dot' operator
#the boston data has dataset, target, features, description and a filename
#let us check the data. the data is shown in terms of arrays.
boston.data
boston.feature_names
#these are the names of the columns
#we can find the information about the data using 'DESCR'
print(boston.DESCR)
#convert the data in to pandas dataframe
dfx = pd.DataFrame(boston.data, columns = boston.feature_names)
#all the independant variables/predictors are named as dfx
dfy = pd.DataFrame(boston.target, columns = ['target'])
#the dependant variable/outcome is the target and it is named as dfy
dfcombine = dfx.join(dfy)
#both the dataframes are combined and named as dfcombine
#let us view and examine the head of the combined dataframe
dfcombine.head()
plt.figure(figsize = (12,6))
sns.heatmap(dfcombine.corr(),annot = True)
#the predictor variable such as crime, INDUS-proportion of non retail business across town,  NOX-nitric oxides concentration 
#(parts per 10 million),Age, RAD -index of accessibility to radial highways, tax, PTRATIO - pupil-teacher ratio by town,
# LSTAT -% lower status of the population have a negative correlation on the target. Increase of any of the above variables
#leads to the decrease in the price of the housing

#the predictor variable such as ZN-proportion of residential land zoned for lots over 25,000 sq.ft.
#, RM-average number of rooms per dwelling , DIS - weighted distances to five Boston employment centres ,
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town, all these variables have good posotove correlation
#with the target. increase in any of the bove variables leads to the increase in the price of the house
#to perform the train test split of the data, the train test split function is imported from sklearn
from sklearn.model_selection import train_test_split
#the percentage of the split is taken as 30%. SO the percentage of training is 70%
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.3, random_state=42)
#the given problem is a classificaton problem. Hence linear regression is used for ML algorithm
from sklearn.linear_model import LinearRegression
linR = LinearRegression()
linR.fit(X_train, y_train)
#the target is predicted for the test dataset
predictions = linR.predict(X_test)
#the accuracy of the prediction is found to be 71% 
linR.score(X_test,y_test)
error = y_test - predictions
#the error is calculated for the above test predictions and a distribution plot is plotted.
sns.distplot(error)
dfx.shape
oness = np.ones((506,1),dtype = int)
dfone = pd.DataFrame(oness, columns = ['ones'])
dfxnew = dfone.join(dfx)
dfxnew.head()
import statsmodels.formula.api as sm
lir_ols = sm.OLS(endog= dfy, exog = dfxnew).fit()
lir_ols.summary()
dfx2 = dfxnew.drop(['INDUS','AGE'], axis = 1)
lir_ols = sm.OLS(endog= dfy, exog = dfx2).fit()
lir_ols.summary()
