%%time

import pandas as pd

import numpy as np

import gc

import os

import random

import glob

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

import warnings 

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl

from matplotlib_venn import venn2

%matplotlib inline



inputPath = '/kaggle/input/used-car-price-forecasting/'

train = pd.read_csv(inputPath + 'train.csv')

test = pd.read_csv(inputPath + 'test.csv')

train['flag'] = 'train'

test['flag'] = 'test'



df = pd.concat([train,test],axis=0)

del train,test

gc.collect()
%%time

# fillna with most frequent value

df['year'].fillna(df['year'].mode()[0], inplace=True)



# fillna with new category

df['model'] = df['model'].fillna('nan')



# fillna with new category

df['condition'] = df['condition'].fillna('nan')



# fillna with new value

df['cylinders'] = df['cylinders'].fillna('-2 cylinders')

df['cylinders'] = df['cylinders'].map(lambda x:x.replace('other','-1 cylinders'))



# fillna with new category

df['fuel'] = df['fuel'].fillna('nan')



# fillna with new value

df['odometer'] = df['odometer'].fillna('-1')

df['odometer'] = df['odometer'].astype(float)



# fillna with new category

df['title_status'] = df['title_status'].fillna('nan')



# fillna with new category

df['transmission'] = df['transmission'].fillna('nan')



# fillna with new category

df['vin'] = df['vin'].fillna('nan')



# fillna with new category

df['drive'] = df['drive'].fillna('nan')



# fillna with new category

df['size'] = df['size'].fillna('nan')



# fillna with new category

df['type'] = df['type'].fillna('nan')



# fillna with new category

df['paint_color'] = df['paint_color'].fillna('nan')
%%time

df['cylinders'] = df['cylinders'].map(lambda x:x.split(' ')[0])

df['cylinders'] = df['cylinders'].astype(int)
%%time

df = pd.get_dummies(df, columns=['paint_color'])
%%time

for c in ['region','manufacturer','model','condition','fuel','title_status','transmission', 'vin', 'drive', 'size', 'type', 'state']:

    lbl = LabelEncoder()

    df[c] = lbl.fit_transform(df[c].astype(str))
%%time

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



def rmse(y_true, y_pred):

    return (mean_squared_error(y_true, y_pred))** .5



train_df = df[df['flag']=='train']

train_df['price'] = np.log1p(train_df['price'])

test_df = df[df['flag']=='test']

drop_features = ['id', 'price', 'description', 'flag']

features = [f for f in train_df.columns if f not in drop_features]



train_x, valid_x, train_y, valid_y = train_test_split(train_df[features], train_df['price'], test_size=0.2, random_state=1,stratify=train_df['manufacturer'])

model = RandomForestRegressor(n_estimators=50,max_depth=10,random_state=1,verbose=1,n_jobs=-1)

model.fit(train_x, train_y)

valid_preds = model.predict(valid_x)

print('Valid RMSE Score:', rmse(valid_y, valid_preds))
importances = model.feature_importances_

indices = np.argsort(importances)[-20:]

plt.figure(figsize=(20, 10))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold



def run_rf_kfold(train_df,test_df,features,target,folds,params):

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])



    cv_list = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[features], train_df['manufacturer'])):

        print ('FOLD:' + str(n_fold))

        

        train_x, train_y = train_df[features].iloc[train_idx], train_df[target].iloc[train_idx]

        valid_x, valid_y = train_df[features].iloc[valid_idx], train_df[target].iloc[valid_idx]

 

        model = params

        model.fit(train_x, train_y)

        oof_preds[valid_idx] = model.predict(valid_x)

        oof_cv = rmse(valid_y,  oof_preds[valid_idx])

        cv_list.append(oof_cv)

        print (cv_list)

        sub_preds += model.predict(test_df[features]) / folds.n_splits

 

    cv = rmse(train_df[target],  oof_preds)

    print('Full OOF RMSE %.6f' % cv)  

    

    train_df['prediction'] = oof_preds

    test_df['prediction'] = sub_preds

    

    return train_df,test_df



train_df = df[df['flag']=='train']

train_df['price'] = np.log1p(train_df['price'])

test_df = df[df['flag']=='test']



target = 'price'

drop_features = ['id', 'price', 'description', 'flag']

features = [f for f in train_df.columns if f not in drop_features]

print ('features:', len(features),features)



n_splits = 5

seed = 817

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

params = RandomForestRegressor(n_estimators=50,max_depth=10,random_state=1,verbose=1,n_jobs=-1)

train_rf,test_rf = run_rf_kfold(train_df,test_df,features,target,folds,params)
import lightgbm as lgb



def run_lgb_kfold(train_df,test_df,features,target,folds,params):

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])



    cv_list = []

    feature_imps = pd.DataFrame()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[features], train_df['manufacturer'])):

        print ('FOLD:' + str(n_fold))

        

        train_x, train_y = train_df[features].iloc[train_idx], train_df[target].iloc[train_idx]

        valid_x, valid_y = train_df[features].iloc[valid_idx], train_df[target].iloc[valid_idx]

 

        dtrain = lgb.Dataset(train_x, label=train_y)

        dval = lgb.Dataset(valid_x, label=valid_y, reference=dtrain) 

        bst = lgb.train(params, dtrain, num_boost_round=10000,

            valid_sets=[dval,dtrain], verbose_eval=500,early_stopping_rounds=100, ) 

        

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)

        oof_cv = rmse(valid_y,  oof_preds[valid_idx])

        cv_list.append(oof_cv)

        print (cv_list)

        sub_preds += bst.predict(test_df[features], num_iteration=bst.best_iteration) / folds.n_splits

 

        feature_imp = pd.DataFrame(sorted(zip(bst.feature_importance('gain'),features)), columns=['Value','Feature'])

        feature_imp['fold'] = n_fold

        feature_imps = pd.concat([feature_imps,feature_imp],axis=0)

        

    cv = rmse(train_df[target],  oof_preds)

    print('Full OOF RMSE %.6f' % cv)  

    

    train_df['prediction'] = oof_preds

    test_df['prediction'] = sub_preds

    

    return train_df,test_df,feature_imps



train_df = df[df['flag']=='train']

train_df['price'] = np.log1p(train_df['price'])

test_df = df[df['flag']=='test']



target = 'price'

drop_features = ['id', 'price', 'description', 'flag']

features = [f for f in train_df.columns if f not in drop_features]

print ('features:', len(features),features)



n_splits = 5

seed = 817

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

params = {

               "objective" : "regression", 

               "boosting" : "gbdt", 

               "metric" : "rmse",  

               "max_depth": -1,

               "min_data_in_leaf": 30, 

               "reg_alpha": 0.1, 

               "reg_lambda": 0.1, 

               "num_leaves" : 31, 

               "max_bin" : 256,

               "learning_rate" :0.2,

               "bagging_fraction" : 0.9,

               "feature_fraction" : 0.9

}

train_lgb,test_lgb,feature_imps = run_lgb_kfold(train_df,test_df,features,target,folds,params)
feature_imp = feature_imps.groupby(['Feature'])['Value'].mean().reset_index()

plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:10])

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()
%%time

for c in ['region','manufacturer','model','condition','fuel','title_status','transmission', 'vin', 'drive', 'size', 'type', 'state']:

    df['count_' + c] = df.groupby([c])['flag'].transform('count')
%%time

df['mean_manufacturer_odometer'] = df.groupby(['manufacturer'])['odometer'].transform('mean')

df['std_manufacturer_odometer'] = df.groupby(['manufacturer'])['odometer'].transform('std')

df['max_manufacturer_odometer'] = df.groupby(['manufacturer'])['odometer'].transform('max')

df['min_manufacturer_odometer'] = df.groupby(['manufacturer'])['odometer'].transform('min')

df['maxmin_manufacturer_odometer'] = df['max_manufacturer_odometer'] - df['min_manufacturer_odometer']
%%time

df['num_chars'] = df['description'].apply(len) 

df['num_words'] = df['description'].apply(lambda x: len(x.split()))

df['num_unique_words'] = df['description'].apply(lambda x: len(set(w for w in x.split())))
train_df = df[df['flag']=='train']

train_df['price'] = np.log1p(train_df['price'])

test_df = df[df['flag']=='test']



target = 'price'

drop_features = ['id', 'price', 'description', 'flag']

features = [f for f in train_df.columns if f not in drop_features]

print ('features:', len(features),features)



n_splits = 5

seed = 817

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

params = {

               "objective" : "regression", 

               "boosting" : "gbdt", 

               "metric" : "rmse",  

               "max_depth": -1,

               "min_data_in_leaf": 30, 

               "reg_alpha": 0.1, 

               "reg_lambda": 0.1, 

               "num_leaves" : 31, 

               "max_bin" : 256,

               "learning_rate" :0.2,

               "bagging_fraction" : 0.9,

               "feature_fraction" : 0.9

}

train_lgb,test_lgb,feature_imps = run_lgb_kfold(train_df,test_df,features,target,folds,params)
feature_imp = feature_imps.groupby(['Feature'])['Value'].mean().reset_index()

plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:10])

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()
params = {

               "objective" : "regression", 

               "boosting" : "gbdt", 

               "metric" : "rmse",  

               "max_depth": -1,

               "min_data_in_leaf": 30, 

               "reg_alpha": 0.1, 

               "reg_lambda": 0.1, 

               "num_leaves" : 31, 

               "max_bin" : 256,

               "learning_rate" :0.1,# 0.2 -> 0.1

               "bagging_fraction" : 0.9,

               "feature_fraction" : 0.9

}

train_lgb,test_lgb,feature_imps = run_lgb_kfold(train_df,test_df,features,target,folds,params)
feature_imp = feature_imps.groupby(['Feature'])['Value'].mean().reset_index()

plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:10])

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()
%%time

test_lgb['price'] = np.expm1(test_lgb['prediction'])

test_df[['id','price']].to_csv('submission.csv',index=False)