# Import Modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
from scipy.special import inv_boxcox
from scipy.stats import boxcox
# read File

Data_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
Data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
Submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
Data_train.head()
Data_test.head()
Submission.head()
Data_train.isnull().sum()
Data_train.info()
Data_test.info()
#Correlation map to see how features are correlated with SalePrice
Cor_heat = Data_train.corr()
plt.figure(figsize=(16,16))
sns.heatmap(Cor_heat, vmax=0.9, square=True)
# Drop Missing value
# I drop these because they had so many missing value and lower corr with sale price

Columns = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

Data_train = Data_train.drop(Columns, axis=1)
Data_train.head()
# Encode Categorical features to numerical features with get dummies from pandas

Data_train = pd.get_dummies(Data_train)
Data_train = Data_train.fillna(method='ffill')
Data_train.head()
## Basic Distribution

print('Skew Value : ' + str(Data_train.SalePrice.skew()))
sns.distplot(Data_train.SalePrice)
Data_train['GrLivArea'] = BCT
# make Models

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
Train = Data_train.drop('SalePrice', axis=1)
Test = Data_train.SalePrice
feature_SP = {'Log Transform': LT,
              'Square Root Transform': SRT,
              'Box-Cox Transform':BCT,
              'Log 1 Transform': L1p}
from sklearn.model_selection import cross_val_score, cross_val_predict

# Perform 5-fold CV

reg = LinearRegression()
lcv = LassoCV()
rcv = RidgeCV()

alg = [reg, lcv, rcv]
    
for y in alg:
    print(str(y) + ' results')
    for key, value in feature_SP.items():
        Test = value
        cvscores_3 = np.sqrt(-cross_val_score(y, Train, Test, scoring='neg_mean_squared_error', cv=5))
        print('RMSE with ' + str(key) + ' : ' + str(np.mean(cvscores_3)))
# Drop Missing value
# I drop these because they had so many missing value and lower corr with sale price

Columns = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

Data_test = Data_test.drop(Columns, axis=1)
Data_test.head()

# Drop Missing Value
Data_test = pd.get_dummies(Data_test)
Data_test = Data_test.fillna(method='ffill')
Data_test.head()
Train = Data_test

# Turn GlivArea into Box Cox
BCT,fitted_lambda = boxcox(Train.GrLivArea,lmbda=None)
Train['GrLivArea'] = BCT

# inv Log Transform
Test = np.exp(LT)
Test = Test[:-1]

# inv Box cox
# Test = inv_boxcox(BCT,fitted_lambda)

result = cross_val_predict(rcv, Train, Test, cv=5)
result
Submission['SalePrice'] = result
Submission.head()
Submission.to_csv("Final Submission File.csv",index=False)