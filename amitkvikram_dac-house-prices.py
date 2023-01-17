# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from IPython.display import display, HTML
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/comp_train.csv")
test = pd.read_csv("../input/comp_test.csv")
print(train.shape)
display(train.head())
print(test.shape)
display(test.head())

# Any results you write to the current directory are saved as output.
numeric_features_train = list(train.select_dtypes(exclude = 
                                                  ['object', 'category']).columns)
numeric_features_test = list(test.select_dtypes(exclude = 
                                               ['object', 'category']).columns)
print("numeric_features_train = ", numeric_features_train)
print("Len = ", len(numeric_features_train))
print("\nnumeric_features_test = ", numeric_features_test)
print("Len = ", len(numeric_features_test))
print(train['SalePrice'].describe())
sns.distplot(train['SalePrice'], kde = True, fit = st.norm)
f = pd.melt(train, id_vars=["SalePrice"], value_vars=numeric_features_train[1:-1])
g = sns.FacetGrid(f, col = "variable", 
                  col_wrap= 2, sharex= False, sharey = False, size = 4)
g.map(sns.distplot, "value", kde = True, fit = st.norm)
train.skew()
print(train.isnull().any())
print(test.isnull().any())
g = sns.FacetGrid(f, col = "variable", col_wrap=2, sharex=False, sharey=False,size=5)
g.map(plt.scatter, "value", "SalePrice")
#POP SalePrice Column
SP = train.pop('SalePrice')
train.head()
#Find theta 
print(train.iloc[:,1:].shape, np.ones((len(train), 1)).shape)
train_X = np.hstack((train.iloc[:,1:],
                       np.ones((len(train), 1))))
SP = SP.values
print(train_X.shape, SP.shape)

#                        train['OverallQual']**2,
#                        train['GrLivArea']**2,
#                        train['GarageArea']**2,
#                        train['YearBuilt']**2,
#                        train['1stFlrSF']**2,
#                        train['TotRmsAbvGrd']**2,

#Split Training and validation data(validation data)
from sklearn.model_selection import train_test_split
def getTheta(r):
    X_train, X_test , y_train, y_test = train_test_split(train_X, SP,
                                                         test_size = 0.2,
                                                        random_state = r)
    #Get theta using normal equation
    theta = np.dot(np.linalg.pinv(X_train), y_train)

    #Error on train and validation data
    print("Training Error =",rmse(predictTarget(X_train, theta), y_train))
    print("Validation Error = ", rmse(predictTarget(X_test, theta), y_test))
    
    return theta
#Predict function which takes theta and X(with 1)
def predictTarget(x, theta):
    return np.dot(x, theta)

#Define RMSE
def rmse(pred, actual):
    assert(len(pred) == len(actual))
    return np.sqrt((((pred - actual)**2).sum())/len(pred))
#Error on train and validation data
# print("Training Error =",rmse(predictTarget(X_train, theta), y_train))
# print("Validation Error = ", rmse(predictTarget(X_test, theta), y_test))
test.head()
#Predict the SALEPRICE
test_X = np.hstack((test.iloc[:,1:],
                       np.ones((len(test), 1))
                   ))

#                        test['OverallQual']**2,
#                        test['GrLivArea']**2,
#                        test['GarageArea']**2,
#                        test['YearBuilt']**2,
#                        test['1stFlrSF']**2,
#                        test['TotRmsAbvGrd']**2,
prediction = np.zeros((test_X.shape[0], 1))
rList = [42, 3, 468, 12, 120, 130, 234, 1000, 100, 200, 1200, 5, 45, 
        49, 300, 320,980, 81, 2100, 1420, 768]
for r in rList:
    y1 = predictTarget(test_X, getTheta(r))
    print(prediction.shape)
    prediction = np.hstack((prediction,y1.reshape(y1.shape[0], 1)))
prediction = np.sum(prediction, axis = 1)/len(rList)
print(test.shape, prediction.shape)
#See characteristics of Predicted Saleprice
pd.DataFrame({'SalePrice': prediction}).describe()
#Save the prediction for submission
sub = pd.DataFrame()
sub['Id'] = test['Id']
sub['SalePrice'] = prediction
sub.to_csv("prediction.csv", index = False)
sub.head()
