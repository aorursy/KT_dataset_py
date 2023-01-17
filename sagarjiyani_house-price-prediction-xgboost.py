# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()
# Check the Skewness
sns.distplot(df['SalePrice']);
df['SalePrice'].skew()
# Remove the Skewness
log_SalePrice = np.log(df['SalePrice'])
log_SalePrice.skew()
sns.distplot(log_SalePrice)
# Correlation map to see how features are correlated with SalePrice
r_mat = df.corr()
sns.heatmap(r_mat)
num = r_mat['SalePrice'].sort_values(ascending=False).head(20).to_frame()
num
total = df.isnull().sum().sort_values(ascending = False)
total.head(20)
# Drop the Unnecessary Columns
df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','MasVnrType'], axis=1 ,inplace=True)
df_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','MasVnrType'], axis=1 ,inplace=True)
# Find Categorical Columns with missing values
cat_mis = df.select_dtypes(include='object').isnull().sum().sort_values(ascending = False)
cat_mis = cat_mis[cat_mis > 0].index
cat_mis
# Filling up missing values with Mode
for col in df[cat_mis]:
    df[col] = df[col].fillna(df[col].mode()[0])
# Find Numerical Columns with missing values
num_mis = df.select_dtypes(exclude='object').isnull().sum().sort_values(ascending = False)
num_mis = num_mis[num_mis > 0].index
# Filling up missing values with Mean
for col in df[num_mis]:
    df[col] = df[col].fillna(df[col].mean())
# Find Categorical Columns with missing values in test data
cat_mis_test = df_test.select_dtypes(include='object').isnull().sum().sort_values(ascending = False)
cat_mis_test = cat_mis_test[cat_mis_test > 0].index
# Fill missing values with mode
for col in df_test[cat_mis_test]:
    df_test[col] = df_test[col].fillna(df_test[col].mode()[0])
# Find Numerical Columns with missing values in test data
num_mis_test = df_test.select_dtypes(exclude='object').isnull().sum().sort_values(ascending = False)
num_mis_test = num_mis_test[num_mis_test > 0].index
# Fill missing values with mean
for col in df_test[num_mis_test]:
    df_test[col] = df_test[col].fillna(df_test[col].mean())
# Outlier
list_of_numerics = df.select_dtypes(exclude='object').columns
types= df.dtypes

outliers= df.apply(lambda x: sum(
                                 (x<(x.quantile(0.25)-1.5*(x.quantile(0.75)-x.quantile(0.25))))|
                                 (x>(x.quantile(0.75)+1.5*(x.quantile(0.75)-x.quantile(0.25))))
                                 if x.name in list_of_numerics else ''))
explo = pd.DataFrame({'Types': types,
                      'Outliers': outliers}).sort_values(by=['Types'],ascending=False)
explo.transpose()
# Removing Outliers
df = df[df['GrLivArea']<4000]
df_test = df_test[df_test['GrLivArea']<4000]
df['MSSubClass'] = df['MSSubClass'].apply(str)
df['YrSold'] = df['YrSold'].astype(str)

df_test['MSSubClass'] = df_test['MSSubClass'].apply(str)
df_test['YrSold'] = df_test['YrSold'].astype(str)
# Extracting the Categorical column from df
categorial_features_train = df.select_dtypes(include=[np.object])
categorial_features_train.head()
# Extracting the Categorical column from df_train
categorial_features_test = df_test.select_dtypes(include=[np.object])
categorial_features_test.head()
# Label Encoding 
encoding = LabelEncoder()
# Train
lbl_enc = {}
for column in categorial_features_train:
    lbl_enc[column] = LabelEncoder()
    df[column] = lbl_enc[column].fit_transform(df[column])
# Test 
lbl_enc = {}
for column in categorial_features_test:
    lbl_enc[column] = LabelEncoder()
    df_test[column] = lbl_enc[column].fit_transform(df_test[column]) 
# Devidingg data into X_train and y_train
X_train = df.drop('SalePrice', axis = 1)
y_train = df['SalePrice']
# XGBoost Regressor
xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
# Fitting the Model
xgb.fit(X_train, y_train)
xgb.score(X_train, y_train)