# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Importing dataset
dataset = pd.read_csv('../input/kc_house_data.csv')
dataset.head(3)
# Check datframe has null values
dataset.isnull().values.any()
dataset.columns
dataset.dtypes
#Correlation of price with all columns
price_corr = dataset[dataset.columns[1:]].corr()['price']
price_corr.sort_values(ascending=False)
#Dropping id and date
dataset = dataset.drop(['id', 'date'], axis=1)
#Plotting correlation matrix
fig, ax = plt.subplots(figsize=(20,20))         
sns.heatmap(dataset.corr(), annot = True)
#understanding the distribution with seaborn
sns.pairplot(dataset[['price', 'sqft_living', 'grade', 'sqft_above', 'sqft_living15', 
                      'bathrooms', 'view']], size=2.5)
plt.show()
# Create method for backward elimination, for choosing perfect model
def backwardElimination(x_opt, y_train, sl) :
    regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
    # If max pvalue is greater than significance level of 5%
    if(max(regressor_OLS.pvalues) > sl):
        index = np.argmax(regressor_OLS.pvalues)
        x_opt = np.delete(x_opt, [index], axis=1)
        x_opt = backwardElimination(x_opt, y_train, sl)
        
    #print(regressor_OLS.summary())    
    return regressor_OLS
# Creating a matrix of features for independent variable, and vector of dependent variable
# Indexes in python start with zero
# Removing last column of dataset, which is dependent variable
x = dataset.loc[:, dataset.columns != 'price'].values
y = dataset.iloc[: , 0].values
print('x shape (matrix of feature): ', x.shape)
print('y shape (vector): ', y.shape)
# Building optimal Model using Backward Elimination
import statsmodels.formula.api as sm
# Stats model api doesnot take into account the intercept b0 in the metrix of features of independent variable
# Add column of 1s for x0, that is 1 for coef b0
# x is 50 rows
# axis = 1 add a column
# Add 1 as end of matrix x, so inverse the arr and values
#x = np.append(arr = x, values = np.ones((21613, 1)).astype(int), axis = 1)

# 1s column will apeear before matrix of features
x = np.append(arr = np.ones((21613, 1)).astype(int), values = x, axis = 1)
print('x shape (matrix of feature): ', x.shape)
print('y shape (vector): ', y.shape)
#splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
# Creating model using backward elimination
x_train_opt = x_train[:,[0,1,2,3,4,5,6,7,8,9,10,
                        11,12,13,14,15,16,17,18]]
regressor = backwardElimination(x_train_opt, y_train, 0.05)
regressor.summary()
# Predicting the test dataset
y_pred = regressor.predict(x_test)
#Predicted value
y_pred
#Actual value
y_test
