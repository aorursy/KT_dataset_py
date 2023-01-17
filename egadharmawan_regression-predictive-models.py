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
f = plt.figure(figsize=(16,16))

# log 1 Transform
ax = f.add_subplot(221)
L1p = np.log1p(Data_train.SalePrice)
sns.distplot(L1p,color='b',ax=ax)
ax.set_title('skew value Log 1 transform: ' + str(np.log1p(Data_train.SalePrice).skew()))

# Square Log Transform
ax = f.add_subplot(222)
SRT = np.sqrt(Data_train.SalePrice)
sns.distplot(SRT,color='c',ax=ax)
ax.set_title('Skew Value Square Transform: ' + str(np.sqrt(Data_train.SalePrice).skew()))

# Log Transform
ax = f.add_subplot(223)
LT = np.log(Data_train.SalePrice)
sns.distplot(LT, color='r',ax=ax)
ax.set_title('Skew value Log Transform: ' + str(np.log(Data_train.SalePrice).skew()))

# Box Cox Transform
ax = f.add_subplot(224)
BCT,fitted_lambda = boxcox(Data_train.SalePrice,lmbda=None)
sns.distplot(BCT,color='g',ax=ax)
ax.set_title('Skew Value Box Cox Transform: ' + str(pd.Series(BCT).skew()))
## Lets see what most important features we have

IF = Cor_heat['SalePrice'].sort_values(ascending=False).head(10).to_frame()
IF.head(4)
# make Models

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# Split The data

Train = Data_train.drop('SalePrice', axis=1)
Test = Data_train.SalePrice
# Assign the distribution of Sale Price

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
    print(str(y.__class__.__name__) + ' results')
    for key, value in feature_SP.items():
        Test = value
        score = np.sqrt(-cross_val_score(y, Train, Test, scoring='neg_mean_squared_error', cv=5))
        print('RMSE with ' + str(key) + ' : ' + str(np.mean(score)))
Columns = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

Data_test = Data_test.drop(Columns, axis=1)
Data_test.head()

# Fill Missing Value
Data_test = pd.get_dummies(Data_test)
Data_test = Data_test.fillna(method='ffill')
Data_test.head()
Train.head()
d1 = Train.columns
d2 = Data_test.columns

out = []

for x in d1:
    if x in d2:
        pass
    else:
        out.append(x)

print(out)
print(len(out))
Data_trains = Train.drop(out, axis=1)
Data_trains.shape
# Use data trains as train
Train = Data_trains
# Best Alg = Ridge CV
model = rcv
# X_train
X_train = Train
# Y_train = Box Cox Sale Price
Test = BCT
y_train = Test
# X_test
X_test = Data_test
# Y_pred
model.fit(X_train, y_train)
y_pred = inv_boxcox(model.predict(X_test), fitted_lambda)
y_pred
Submission['SalePrice'] = y_pred
Submission.head()
Submission.to_csv("Final Submission File.csv",index=False)