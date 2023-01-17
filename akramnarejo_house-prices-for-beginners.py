# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from scipy.stats import skew, kurtosis

%matplotlib inline

sns.set()
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# checking the shape of both datasetes

print(train.shape)

print(test.shape)
target = train[['SalePrice']]

# dropping the target variable from train dataset

train.drop(columns=['SalePrice'],axis=1, inplace=True) 
# Now concating both datasets

df = pd.concat([train,test])

df.shape
df.info()
df.drop(columns=['Id'], axis=1, inplace=True)

df.shape
plt.figure(figsize=(16,12))

msno.bar(df, labels=True, fontsize=(10))
#categorical variables having missing values more than 30%

for col in df:

    if df[col].dtype == 'object':

        if (df[col].isnull().sum()/df[col].isnull().count())*100 >=30:

            print(col)
df.drop(columns=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)
for col in df:

    if df[col].dtype == 'object':

        df[col] = df[col].fillna(df[col].mode()[0])
# Now for numericals

check = False

for col in df:

    if df[col].dtype != 'object':

        if (df[col].isnull().sum()/df[col].isnull().count())*100 >=30:

            print(col)

        else:

            check = True

if check:

    print('We do not have missing data more than 30%')
# let's see how much data we have missing in numerical data

for col in df:

    if df[col].dtype != 'object':

        print(df[col].isnull().sum(), end=' ')
for col in df:

    if df[col].dtype != 'object':

        df[col] = df[col].fillna(0)
plt.figure(figsize=(16,12))

msno.bar(df, labels=True, fontsize=(10))
df = pd.get_dummies(df)

df.shape
from sklearn.preprocessing import MinMaxScaler

df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df))

df_scaled.columns = df.columns

df_scaled
train = df_scaled.iloc[:1460,:]

test = df_scaled.iloc[1460:,:]

print(train.shape)

print(test.shape)
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(train,target,test_size=0.2,random_state=1)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)

lr_preds = lr.predict(X_val)
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=1)

dt.fit(X_train,y_train)

dt_preds = dt.predict(X_val)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=1)

rf.fit(X_train,y_train)

rf_preds = rf.predict(X_val)
from xgboost import XGBRegressor

xgb = XGBRegressor(random_state=1)

xgb.fit(X_train,y_train)

xgb_preds = xgb.predict(X_val)
from sklearn.metrics import mean_absolute_error
compare = {'Models':['LinearRegression','DecissionTree','RandomForest','XGBoost'],

          'MeanAbsoluteError':[mean_absolute_error(lr_preds,y_val), mean_absolute_error(dt_preds,y_val), mean_absolute_error(rf_preds,y_val),

                              mean_absolute_error(xgb_preds,y_val)]}

pd.DataFrame(compare)
test_values = xgb.predict(test)

test_values
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission['SalePrice'] = test_values

submission
submission.to_csv('Submission_3.csv', index=False)