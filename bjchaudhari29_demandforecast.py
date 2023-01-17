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
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold,train_test_split,KFold
import re
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,f1_score

train_df = pd.read_csv('/kaggle/input/train_0irEZ2H.csv')
test_df = pd.read_csv('/kaggle/input/test_nfaJ3J5.csv')
submission = pd.read_csv('/kaggle/input/sample_submission_pzljTaX.csv')
train_df = train_df.dropna(axis=0, subset=['total_price'])
train_df['train_or_test']='train'
test_df['train_or_test']='test'
df=pd.concat([train_df,test_df])
print(train_df.isnull().sum())
print(test_df.dtypes)
train_df['week']= pd.to_datetime(train_df['week'])
test_df['week']= pd.to_datetime(test_df['week'])
df=pd.concat([train_df,test_df])
!pip install pendulum
def create_date_featues(df):

    df['Year'] = pd.to_datetime(df['week']).dt.year

    df['Month'] = pd.to_datetime(df['week']).dt.month

    df['DayOfyear'] = pd.to_datetime(df['week']).dt.dayofyear

    df['Week'] = pd.to_datetime(df['week']).dt.week

    #df['Quarter'] = pd.to_datetime(df['week']).dt.quarter 

    df['Is_month_start'] = pd.to_datetime(df['week']).dt.is_month_start

    df['Is_month_end'] = pd.to_datetime(df['week']).dt.is_month_end
    
    
    
    #df['Is_quarter_start'] = pd.to_datetime(df['week']).dt.is_quarter_start

    #f['Is_quarter_end'] = pd.to_datetime(df['week']).dt.is_quarter_end

    #df['Semester'] = np.where(df['week'].isin([1,2]),1,2)

    return df
df=create_date_featues(df)
df['new_feat1'] = df['total_price'] - df['base_price']
df['More'] = np.where(df['new_feat1'] < 0, 1, 0)
df['less'] = np.where(df['new_feat1'] > 0, 1, 0)
df['same'] = np.where(df['new_feat1'] == 0, 1, 0)
for col in ['sku_id','store_id']:
    df = pd.get_dummies(df, columns=[col])
df['latest'] = df['store_id'] + df['sku_id']
for col in ['latest']:
    df = pd.get_dummies(df, columns=[col])
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
#df['sku_id'] = labelencoder.fit_transform(df['sku_id'])
#df['store_id'] = labelencoder.fit_transform(df['store_id'])
df['latest'] = labelencoder.fit_transform(df['latest'])
plt.figure(figsize=(24, 6))
plt.subplot(121)
sns.distplot(df["total_price"])
plt.subplot(122)
sns.distplot(np.log1p(df["total_price"]))
plt.show()
df['total_price']=np.log1p(df["total_price"])
plt.figure(figsize=(24, 6))
plt.subplot(121)
sns.distplot(df["base_price"])
plt.subplot(122)
sns.distplot(np.log1p(df["base_price"]))
plt.show()
df['base_price']=np.log1p(df["base_price"])
plt.figure(figsize=(24, 6))
plt.subplot(121)
sns.distplot(train_df["units_sold"])
plt.subplot(122)
sns.distplot(np.log1p(train_df["units_sold"]))
plt.show()
train=df.loc[df.train_or_test.isin(['train'])]
test=df.loc[df.train_or_test.isin(['test'])]
train.drop(columns={'train_or_test'},axis=1,inplace=True)
test.drop(columns={'train_or_test'},axis=1,inplace=True)
train['units_sold']=np.log1p(train['units_sold'])
y = train.units_sold
np.expm1(y)
train=train.drop(columns={'record_ID','week','units_sold'},axis=1)
test=test.drop(columns={'record_ID','week','units_sold'},axis=1)
train=train.drop(columns={'record_ID','week','units_sold','total_price', 'base_price'},axis=1)
test=test.drop(columns={'record_ID','week','units_sold','total_price', 'base_price'},axis=1)
print(train.shape,test.shape)
X_train, X_test, Y_train, Y_test = train_test_split(train, y, test_size = 0.1, random_state = 5)
features = [col for col in train.columns]
cat = []
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from math import sqrt
import keras

param = {'boosting_type': 'gbdt','num_leaves':128, 'objective':'regression','max_depth':6,'eval_metric':'rmse','learning_rate':.2,'max_bin':100}
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1048)
kf=StratifiedKFold(n_splits=3, shuffle=True, random_state=999)
num_classes = len(np.unique(y))
y_train_categorical = keras.utils.to_categorical(y, num_classes)
predictions = np.zeros((len(test), ))
feature_importance_df = pd.DataFrame()
#for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, y.values)):
for fold_, (trn_idx, val_idx) in enumerate(kf.split(train, y_train_categorical.argmax(1))):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=y.iloc[trn_idx],categorical_feature=cat)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=y.iloc[val_idx],categorical_feature=cat)

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 500)
    predictions_val = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    #print("MSE: {:<8.5f}".format(mean_squared_error(predictions_val, y.iloc[val_idx])))
    #print("RMSLE: {:f}".format(np.sqrt(mean_squared_log_error(np.expm1(y.iloc[val_idx]),np.expm1(predictions_val)))))
    #print("RMSLE: {:f}".format(np.sqrt(mean_squared_error(y.iloc[val_idx], predictions_val))))
    def rmsle(y_true, y_pred):
        assert len(y_true) == len(y_pred)
        return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))
    print(rmsle(np.expm1(y.iloc[val_idx]),np.expm1(predictions_val)))
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
nas_lgb =np.expm1(predictions)
from xgboost import XGBRegressor
folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=1048)
kf=StratifiedKFold(n_splits=2, shuffle=True, random_state=999)
num_classes = len(np.unique(y))
y_train_categorical = keras.utils.to_categorical(y, num_classes)
predictions = np.zeros((len(test), ))
feature_importance_df = pd.DataFrame()
#for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, y.values)):
for fold_, (trn_idx, val_idx) in enumerate(kf.split(train, y_train_categorical.argmax(1))):
    print("Fold {}".format(fold_))
    #trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=y.iloc[trn_idx])
    #val_data = lgb.Dataset(train.iloc[val_idx][features], label=y.iloc[val_idx])
    model = XGBRegressor(
    max_depth=8,
    n_estimators=10000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,
    seed=42)
    #num_round = 1000000
    #clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 500)
    #predictions_val = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    model.fit(train.iloc[trn_idx][features], y.iloc[trn_idx],eval_metric="rmse", 
    eval_set=[(train.iloc[trn_idx][features], y.iloc[trn_idx]), (train.iloc[val_idx][features], y.iloc[val_idx])], 
    verbose=1000, 
    early_stopping_rounds = 100)
    predictions_val=model.predict(train.iloc[val_idx][features])
    def rmsle(y_true, y_pred):
        assert len(y_true) == len(y_pred)
        return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))
    print(rmsle(np.expm1(y.iloc[val_idx]),np.expm1(predictions_val)))
    
    predictions += model.predict(test[features]) / folds.n_splits


nas_xgb =np.expm1(predictions)
ens = nas_xgb* 0.6 + nas_lgb* 0.4
final_dict = {'record_ID' : submission.record_ID, 'units_sold': ens}
Result = pd.DataFrame(final_dict)
Result.head()
Result.to_csv('finalsub.csv',index=False)
fold_importance_df.sort_values('importance',ascending=False).head(50)
#df.sort_values('2')


