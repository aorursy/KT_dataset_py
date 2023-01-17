# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
semple = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
#desc = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt',sep=" ",header=None)
df = train.append(test, sort=False, ignore_index=True)

df.info()
null = df.isnull().sum()*100/len(df)
null[null>0]

no_object = [i for i in df.columns if df.dtypes[i] != 'object']
null1 = df[no_object].isnull().sum()*100/len(df[no_object])
null1[null1>0]
for i in ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars',
            'GarageArea']:
    df[i] = df[i].fillna(0)
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].apply(
    lambda x: x.fillna(x.median()))
df["GarageYrBlt"] = df.groupby("Neighborhood")["GarageYrBlt"].apply(
    lambda x: x.fillna(x.median()))
object_ = [i for i in df.columns if df.dtypes[i] == 'object']
null2 = df[object_].isnull().sum()*100/len(df[object_])
null2[null2>0]
for i in ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish',
           'GarageQual','GarageCond','Alley','Fence','PoolQC','MiscFeature']:
    df[i] = df[i].fillna('None')


df['MSZoning'] = df['MSZoning'].fillna('RL')
df['Utilities'] = df['Utilities'].fillna('AllPub')
df['Exterior1st'] = df['Exterior1st'].fillna('VinylSd ')
df['Exterior2nd'] = df['Exterior2nd'].fillna('VinylSd')
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['Electrical'] = df['Electrical'].fillna('SBrk')
df['KitchenQual'] = df['KitchenQual'].fillna('TA')
df['Functional'] = df['Functional'].fillna('Typ')
df['SaleType'] = df['SaleType'].fillna('WD')
df.isnull().sum()
len(no_object )
train['MSSubClass']
fig, ax = plt.subplots(19, 2,figsize=(15,25))
fig.subplots_adjust(hspace=1.5)
for i, b in zip(no_object,ax.flatten()):
       sns.boxplot(df[i], ax = b)
       
def outlier(df):
    for col in df.columns:
        if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
            percentiles = df[col].quantile([0.25,0.75]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
        else:
            df[col]=df[col]
    return df

df = outlier(df)
df.head()
df.info()
"""def outliers(df,n,features):
    
    outlier= []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        outlier_IQR = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list = df[(df[col] < Q1 - outlier_IQR) | (df[col] > Q3 + outlier_IQR )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier.extend(outlier_list)
        
    # select observations containing more than 2 outliers
    outlier = Counter(outlier)        
    multi = list( k for k, v in outlier.items() if v > n )
    
    return multi  

Outliers_drop = outliers(df,2, no_object)
df.loc[Outliers_drop] # Show the outliers rows"""

#df = df.drop(Outliers_drop, axis = 0).reset_index(drop=True)
no_object = [i for i in df.columns if df.dtypes[i] != 'object']
no_object 
df
from sklearn.preprocessing import LabelEncoder

for i in object_:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])
df.shape
#creating matrices for feature selection:
X_train = df[:train.shape[0]]
X_test_fin = df[train.shape[0]:]
y = train.SalePrice
X_train['Y'] = y
df = X_train


X = df.drop('Y', axis=1)
y = df.Y
X_test_fin.shape,y.shape, X.shape, y.shape
#tu dale dela vse ok
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

"""
X = df.drop('Y', axis=1)
y = df.Y"""



params = {
        'objective':'reg:linear',
        'n_estimators': 10,
        'booster':'gbtree',
        'max_depth':2,
        'eval_metric':'rmse',
        'learning_rate':0.1, 
        'min_child_weight':1,
        'subsample':0.7,
        'colsample_bytree':0.81,
        'seed':45,
        'reg_alpha':1e-05,
        'gamma':0,
        'nthread':-1

}


x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(X_test_fin)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 1200, watchlist, early_stopping_rounds=100, maximize=False, verbose_eval=10)

p_test = clf.predict(d_test)
_
sub = pd.DataFrame()
sub['Id'] = test['Id']
sub['SalePrice'] = p_test
#sub['SalePrice'] = sub.apply(lambda r: leaks[int(r['Id'])] if int(r['Id']) in leaks else r['SalePrice'], axis=1)
sub.to_csv('submission_pred_quantil6.csv', index=False)
from sklearn.metrics import mean_absolute_error, r2_score
d_test = clf.predict(d_valid)
r2_score(y_valid, d_test)
sub.shape
q1 = sub['SalePrice'].quantile(0.0045)
q2 = sub['SalePrice'].quantile(0.99)
sub['SalePrice'] = sub['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
sub['SalePrice'] = sub['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
sub.to_csv("submission_oboje_quantile6.csv", index=False)
