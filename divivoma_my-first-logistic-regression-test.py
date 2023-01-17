

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import math as mat



from scipy import stats

from scipy.stats import norm

#from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

#import statsmodels.formula.api as smf

import statsmodels.api as sm

from patsy import dmatrices



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# Any results you write to the current directory are saved as output.



import sklearn.linear_model as LinReg

import sklearn.linear_model as LogReg

import sklearn.metrics as metrics



#from scipy.optimize import fmin_bfgs



#loading the data 

data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
data_train.shape
# visualize the relationship between some features and the Sale price using scatterplots

#sns.pairplot(data_train, x_vars=['GrLivArea','TotalBsmtSF'], y_vars='SalePrice', size=7, aspect=0.7)
vars = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath','YearBuilt']

Y = data_train[['SalePrice']] #dim (1460, 1)

ID_train = data_train[['Id']] #dim (1460, 1)

ID_test = data_test[['Id']]   #dim (1459, 1)

#extract only the relevant feature with cross correlation >0.5 respect to SalePrice

X_matrix = data_train[vars]

X_matrix.shape  #dim (1460,6)



X_test = data_test[vars]

X_test.shape   #dim (1459,6)
#check for missing data:

#missing data

total = X_matrix.isnull().sum().sort_values(ascending=False)

#check = houses.isnull().count() #this gives the number of elements for each column (feature)



percent = (X_matrix.isnull().sum()/X_matrix.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)



#no missing data in this training set


#substitute NA data with the mean value of that feature:

X_test['TotalBsmtSF']=X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].mean())

X_test['GarageCars']=X_test['GarageCars'].fillna(mat.ceil(X_test['GarageCars'].mean()))



#let's drop NA value from the matrix:

#X_test = X_test.dropna()





total = X_test.isnull().sum().sort_values(ascending=False)

#check = houses.isnull().count() #this gives the number of elements for each column (feature)



percent = (X_test.isnull().sum()/X_test.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)

X_test.shape #now the dimensions are (1457, 6) 

max_abs_scaler = preprocessing.MaxAbsScaler()

X_train_maxabs = max_abs_scaler.fit_transform(X_matrix)

print(X_train_maxabs)
X_test_maxabs = max_abs_scaler.fit_transform(X_test)

print(X_test_maxabs)
lr=LinReg.LinearRegression().fit(X_train_maxabs,Y)



Y_pred_train = lr.predict(X_train_maxabs)

print("Lin Reg performance evaluation on Y_pred_train")

print("R-squared =", metrics.r2_score(Y, Y_pred_train))

#print("Coefficients =", lr.coef_)



Y_pred_test = lr.predict(X_test_maxabs)

print("Lin Reg performance evaluation on X_test")

#print("R-squared =", metrics.r2_score(Y, Y_pred_test))

print("Coefficients =", lr.coef_)