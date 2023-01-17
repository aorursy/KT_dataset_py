# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# importing the dataset
dataset = pd.read_csv('../input/winequality-red.csv')
dataset.head()
# getting the summary of the data
# as we can see there is huge difference between 75% percentile and max values for residual sugar, free sulfur dioxide and total sulfur
# dioxide - which means there are some huge outliers in the data
dataset.describe()
# let's build a correlation matrix and use seaborn to plot the heatmap of these
# correlation matrix
corr_matrix = dataset.corr()
sns.heatmap(corr_matrix,cmap="YlGnBu")
# from the heatmap, dark shades represent positive correlation and light shades represent negative correlation
# we can see that "fixed acidity" and "citric acid", "density" and "fixed acidity", 
# "free sulfur dioxide" and "total sulfur dioxide" are highly correlated

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr_matrix,cmap="YlGnBu", annot=True, linewidths=.5, ax = ax)
# I will use linear regression model to determine the quality of the wine
# I will use backward elimination method
# Steps are as follows :
# 1. select the signifcance level to stay in the model (eg : SL = 0.05)
# 2. Fit the model to all the possible predictors
# 3. Consider the predictor with highest P-value, if P > SL then go to STEP 4 or else to STEP 6
# 4. Remove the predictor
# 5. Fit the model without this variable and go to step 3
# 6. Finish the model

# let us remove the multicollinear variables from the dataset
linear_dataset = dataset.drop(['fixed acidity','density','citric acid','free sulfur dioxide','total sulfur dioxide'],axis = 1,inplace=False)

X = linear_dataset.loc[:,'volatile acidity' : 'alcohol'].values
y = linear_dataset.loc[:, 'quality'].values
# applying backward elimination
X_opt = X[:,0:6]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# now looking at the P-value column(P>|t|) we have to eliminate the columns whose P-value SL(0.05)
# we have to remove x2 column
X_opt = X[:,[0,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# that's it. There is no more predictor with p-value > SL so will stop here 
# and we can see that R-squared and Adj.R-squared are very close to 1
# our model fits well
# now rebuilding our training and test dataset and predicting the result
# after backward elimination we found that few predictors to be removed
# so let us bulid a new training and test data set from these observations


X_train,X_test,y_train,y_test = train_test_split(X_opt,y,test_size = 0.2, random_state = 0)
# feature scaling to get optimized result
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# fitting linear regression to training set
regressor = LinearRegression()
regressor.fit(X_train,y_train)
# calulating the RMSE value to see how well the model predicts
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, y_pred))
rmse