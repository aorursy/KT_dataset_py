import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

#libraries

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



from sklearn.metrics import r2_score

df_train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

submission=pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
df_train.head()
df_test.head()
submission.head()
df_train=df_train.drop('Id',axis=1)
{'train shape':df_train.shape,'test shape':df_test.shape}
np.unique(df_train.dtypes)
df_train.loc[:,df_train.dtypes=='int64'].head()
df_train.loc[:,(df_train.dtypes=='float64')].head()
df_train.loc[:,(df_train.dtypes=='O')].head()
columna=[]

n_missing=[]

for column in df_train.columns:

    columna.append(column)

    n_missing.append(df_train[column].isna().sum()/len(df_train[column]))

missing=pd.DataFrame({'column':columna,'n_missing':n_missing}).sort_values('n_missing',ascending=False).head(20)

sns.set_color_codes("pastel")

sns.barplot(x="n_missing", y="column", data=missing,

            label="missing", color="b")
columna=[]

n_missing=[]

for column in df_test.columns:

    columna.append(column)

    n_missing.append(df_test[column].isna().sum()/len(df_test[column]))

missing=pd.DataFrame({'column':columna,'n_missing':n_missing}).sort_values('n_missing',ascending=False).head(20)

sns.set_color_codes("pastel")

sns.barplot(x="n_missing", y="column", data=missing,

            label="missing", color="b")
for column in ["PoolQC","Alley","Fence","FireplaceQu",

               "GarageType","GarageCond","GarageQual",

               "GarageFinish","BsmtQual","BsmtCond",

               "BsmtExposure","BsmtFinType1","BsmtFinType2",

               "MiscFeature","MasVnrType"]:

    df_train[column] = df_train[column].fillna("None")

    df_test[column] = df_test[column].fillna("None")

{'shape train':df_train.shape,'shape test':df_test.shape}
columna=[]

n_missing=[]

for column in df_train.columns:

    columna.append(column)

    n_missing.append(df_train[column].isna().sum()/len(df_train[column]))

missing1=pd.DataFrame({'column':columna,'n_missing':n_missing}).sort_values('n_missing',ascending=False).head(5)



sns.set_color_codes("pastel")

sns.barplot(x="n_missing", y="column", data=missing1,

            label="missing", color="b")
from sklearn import preprocessing

labels=['ExterQual',

'ExterCond',

'BsmtQual',

'BsmtCond',

'BsmtExposure',

'BsmtFinType1',

'BsmtFinType2',

'HeatingQC',

'CentralAir',

'KitchenQual',

'Functional',

'FireplaceQu',

'GarageFinish',

'GarageQual',

'GarageCond',

'PavedDrive',

'PoolQC',

'Fence']



for i in labels:

    le = preprocessing.LabelEncoder()

    df_train[i] = df_train[i].fillna("None")

    df_test[i] = df_test[i].fillna("None")

    le.fit(np.unique(np.concatenate([df_test[i].unique(),df_train[i].unique()])))

    df_train[i]=le.transform(df_train[i])

    df_test[i]=le.transform(df_test[i])
df_test=pd.get_dummies(df_test, columns=df_train.loc[:,(df_train.dtypes=='O')].columns)

df_train=pd.get_dummies(df_train, columns=df_train.loc[:,(df_train.dtypes=='O')].columns)

for column in df_train.columns[~df_train.columns.isin(df_test.columns)]:

    df_test[column]=0
columna=[]

n_missing=[]

for column in df_test.columns:

    columna.append(column)

    n_missing.append(df_test[column].isna().sum()/len(df_test[column]))

missing2=pd.DataFrame({'column':columna,'n_missing':n_missing}).sort_values('n_missing',ascending=False).head(15)



sns.set_color_codes("pastel")

sns.barplot(x="n_missing", y="column", data=missing2,

            label="missing", color="b")
from sklearn.linear_model import LinearRegression
## train

target_col1=missing1.column.values[[0,1,2]]



## test

target_col2=missing2.column.values[[0,1,2,3,4,5,6,7,8,9,10,11]]
for i in [0,1,2]:

    X_train1=df_train.drop(target_col1,axis=1)

    X_test1=X_train1[df_train[target_col1[i]].isna()]

    X_train1=X_train1[~df_train[target_col1[i]].isna()]

    y_train1=df_train[target_col1[i]][~df_train[target_col1[i]].isna()]

    regr = LinearRegression()

    regr.fit(X_train1, y_train1)

    df_train.loc[df_train[target_col1[i]].isna(),target_col1[i]]=regr.predict(X_test1)
for i in range(11):

    X_train2=df_test.drop(target_col2,axis=1)

    X_test2=X_train2[df_test[target_col2[i]].isna()]

    X_train2=X_train2[~df_test[target_col2[i]].isna()]

    y_train2=df_test[target_col2[i]][~df_test[target_col2[i]].isna()]

    X_test2.head()

    regr =LinearRegression()

    regr.fit(X_train2.drop('Id',axis=1), y_train2)

    df_test.loc[df_test[target_col2[i]].isna(),target_col2[i]]=regr.predict(X_test2.drop('Id',axis=1))
df_test
X_train, X_test, y_train,test_id=df_train.drop(['SalePrice'], axis=1),df_test.drop(['Id','SalePrice'], axis=1),df_train[['SalePrice']],df_test[['Id']]
X_train, X_val, y_train, y_val=train_test_split(

    X_train,

    y_train,

    shuffle=True,

    test_size=0.2,

    random_state=41)
{'shape train':X_train.shape,'shape test':X_test.shape,'shape val':X_val.shape}
mm_scaler = preprocessing.MinMaxScaler()

X_train = pd.DataFrame(mm_scaler.fit_transform(X_train))

X_val = pd.DataFrame(mm_scaler.transform(X_val))

X_test = pd.DataFrame(mm_scaler.transform(X_test))
def root_mean_squared_log_error(y_valid, y_preds):

    """Calculate root mean squared error of log(y_true) and log(y_pred)"""

    if len(y_preds)!=len(y_valid): return 'error_mismatch'

    y_preds_new = [math.log(x) for x in y_preds]

    y_valid_new = [math.log(x) for x in y_valid]

    return mean_squared_error(y_valid_new, y_preds_new, squared=False)
# Light Gradient Boosting Regressor

lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=7000,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=4, 

                       bagging_seed=8,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)



# Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)  



# Random Forest Regressor

rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          random_state=42)
lightgbm.fit(X_train,y_train)

pred_val=lightgbm.predict(X_val)

pred_test=lightgbm.predict(X_test)

print(root_mean_squared_log_error(y_val.values, pred_val))

print(r2_score(y_val.values, pred_val))
rf.fit(X_train,y_train)

pred_val=rf.predict(X_val)

pred_test=rf.predict(X_test)

print(root_mean_squared_log_error(y_val.values, pred_val))

print(r2_score(y_val.values, pred_val))
gbr.fit(X_train,y_train)

pred_val=gbr.predict(X_val)

pred_test=gbr.predict(X_test)

print(root_mean_squared_log_error(y_val.values, pred_val))

print(r2_score(y_val.values, pred_val))
X_train, X_test, y_train,test_id=df_train.drop(['SalePrice'], axis=1),df_test.drop(['Id','SalePrice'], axis=1),df_train[['SalePrice']],df_test[['Id']]
gbr.fit(X_train,y_train)

rf.fit(X_train,y_train)

lightgbm.fit(X_train,y_train)

pred_test=(gbr.predict(X_test)+rf.predict(X_test)+lightgbm.predict(X_test))/3
sub=pd.DataFrame({'Id':test_id.Id,'SalePrice':np.array(pred_test)})

sub
submission=sub.merge(submission.drop('SalePrice',axis=1),on='Id')
submission.to_csv('submission.csv', index=False)