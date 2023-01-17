# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_train.shape
df_train.info()
df_train.head()
df_out = df_train['target'].sort_values().reset_index(drop=True)

df_out.head()

#df_out.unique()

#drop= true removes index column
plt.scatter(x=df_out.index, y=df_out.values)

plt.show()
plt.scatter(x=df_out.index, y=np.log1p(df_out.values))

plt.show()
fig, ax = plt.subplots(figsize=(8, 8))

sns.distplot(df_train['target'])

ax.set_xlabel('Index', fontsize=12)

ax.set_ylabel('Target', fontsize=12)

ax.set_title('Distribution of Target', fontsize=14)

plt.show()
fig, ax = plt.subplots(figsize=(8, 8))

sns.distplot(np.log1p(df_train['target']))

ax.set_xlabel('Index', fontsize=12)

ax.set_ylabel('Target', fontsize=12)

ax.set_title('Distribution of Log of Target', fontsize=14)

plt.show()
df_test=pd.read_csv('../input/test.csv')
df_test.head()
df_test.info()
#pd.concat([df_train,df_test]).drop_duplicates(keep=False)

df_train.columns.difference(df_test.columns)
df_null = df_train.isna().sum().reset_index()

df_null.columns = ['Column_Name', 'Count']
df_null = df_null[df_null['Count'] > 0]

df_null
df_null_test=df_test.isna().sum().reset_index()

df_null_test.columns=['Column_Name', 'Null Count']
df_null_test= df_null_test[df_null_test['Null Count']>0]

df_null_test
df_unique = df_train.nunique().reset_index()

df_unique.columns = ['Column_Name', 'Unique_Count']

df_unique.head()
df_unique = df_unique[df_unique['Unique_Count'] == 1]

df_unique.head()# 256 columns
unique_cols_train = df_unique[df_unique['Unique_Count'] == 1]['Column_Name'].values

len(unique_cols_train)
df_test_unique=df_test.nunique().reset_index()

df_test_unique.columns=['Column_Name', 'Unique_count']
unique_cols_test = df_test_unique[df_test_unique['Unique_count']==1]['Column_Name'].values

len(unique_cols_test) # no columns
unique_list_test=df_test[unique_cols_train]
df_unique_test= unique_list_test.nunique().reset_index()

df_unique_test.columns=['Column_Name','Unique_Count']

df_unique_test.head()
pd.merge(df_unique, df_unique_test, on='Column_Name').head()
df_train.drop(unique_cols_train, axis = 1, inplace=True)
df_test.drop(unique_cols_train, axis=1, inplace=True)
df_train.shape, df_test.shape
X_train = df_train.iloc[:, 2:]

Y_train = df_train.iloc[:, 1]
X_train.shape, Y_train.shape, df_test.shape
X_test =  df_test.iloc[:, 1:]
X_test.shape
from scipy.stats import spearmanr

corr_list = []

for cols in X_train.columns:

    corr_list.append([cols, spearmanr(a=X_train[cols], b=Y_train)[0]])

df_corr = pd.DataFrame(corr_list, columns=['Column_Name', 'Correlation'])
df_corr = df_corr.sort_values(by='Correlation').reset_index(drop=True)

df_corr.head()
df_corr = df_corr[(df_corr['Correlation'] < -0.11) | (df_corr['Correlation'] > 0.1)]

df_corr.shape
fig, ax = plt.subplots(figsize=(8,8))

sns.barplot(x='Correlation', y='Column_Name', data=df_corr, color='grey')

ax.set_xlabel('Correlation', fontsize=12)

ax.set_ylabel('Column', fontsize=12)

plt.show()
from sklearn.model_selection import train_test_split

X_dev, X_val, y_dev, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
X_dev.shape, X_val.shape, y_dev.shape, y_val.shape
from sklearn.model_selection import GridSearchCV



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

import lightgbm as lgb

from sklearn.metrics import mean_squared_log_error
'''rfr=RandomForestRegressor()

rf_parm={   "n_estimators": [100, 300, 500, 1000],

    "bootstrap": [True, False],

    "max_depth": [2, 4, 6],

    "max_features": ['sqrt', 'log2'],

    "min_samples_split": [2, 4, 6],

    "min_samples_leaf": [2, 4, 6]

}



grid_search=GridSearchCV(estimator=rfr, param_grid=rf_parm, cv=4)

grid_search.fit(X_dev,y_dev)



grid_search.best_estimator_'''
# Best score for Random Forest Regressor

model_rf=RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=6,

           max_features='sqrt', max_leaf_nodes=None,

           min_impurity_decrease=0.0, min_impurity_split=None,

           min_samples_leaf=2, min_samples_split=2,

           min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None,

           oob_score=False, random_state=None, verbose=0, warm_start=False)
model_lgb= lgb.LGBMRegressor(

        objective= 'regression',num_leaves= 58,subsample= 0.6143,colsample_bytree= 0.6453,

        min_split_gain= np.power(10, -2.5988),reg_alpha= np.power(10, -2.2887),reg_lambda= np.power(10, 1.7570),

        min_child_weight= np.power(10, -0.1477),verbose= -1,seed= 3,boosting_type= 'gbdt',max_depth= -1,learning_rate= 0.05,metric= 'l2')
model_xgb=xgb.XGBRegressor(

n_estimators=2000,

max_depth=4,

min_child_weight=2,

gamma=0.9,

subsample=0.8,

colsample_bytree=0.8,

n_thread=-1,

scale_pos_weight=1)
model_gbm = GradientBoostingRegressor(

              criterion='friedman_mse', init=None,

              learning_rate=0.1,  max_depth=2,

              max_features='log2', max_leaf_nodes=None,

              min_impurity_decrease=0.0, min_impurity_split=None,

              min_samples_leaf=2, min_samples_split=6,

              min_weight_fraction_leaf=0.0, n_estimators=100,

              n_iter_no_change=None, presort='auto', random_state=None,

              subsample=1.0, tol=0.0001, validation_fraction=0.1,

              verbose=0, warm_start=True)
model_rf.fit(X_dev, y_dev)

model_lgb.fit(X_dev, y_dev)

model_xgb.fit(X_dev, y_dev)

model_gbm.fit(X_dev, y_dev)
rf_feature_rank= model_rf.feature_importances_

lgb_feature_rank= model_lgb.feature_importances_

xgb_feature_rank= model_xgb.feature_importances_

gbm_feature_rank= model_gbm.feature_importances_
df_important_features=pd.DataFrame({

    "Features" : X_dev.columns,

    "RF_Col_Rank" : rf_feature_rank,

    "LGB_Col_Rank" : lgb_feature_rank,

    "XGB_Col_Rank" : xgb_feature_rank,

    "gbm_Col_Rank" : gbm_feature_rank,

})
df_important_features.head()
fig, ax=plt.subplots(figsize=(8,10))

sns.barplot(x='RF_Col_Rank', y='Features', data=df_important_features.iloc[:30, :])

ax.set_title('Features in the order of importance for Random forest Regressor', fontsize=12)

ax.set_xlabel('Rank', fontsize=12)

ax.set_ylabel('Columns/Features', fontsize=12)

plt.show()
fig, ax=plt.subplots(figsize=(8,10))

sns.barplot(x='LGB_Col_Rank', y='Features', data=df_important_features.iloc[:30, :])

ax.set_title('Features in the order of importance for Light GBM Regressor', fontsize=12)

ax.set_xlabel('Rank', fontsize=12)

ax.set_ylabel('Columns/Features', fontsize=12)

plt.show()
fig, ax=plt.subplots(figsize=(8,10))

sns.barplot(x='XGB_Col_Rank', y='Features', data=df_important_features.iloc[:30, :])

ax.set_title('Features in the order of importance for XG Boost Regressor', fontsize=12)

ax.set_xlabel('Rank', fontsize=12)

ax.set_ylabel('Columns/Features', fontsize=12)

plt.show()
fig, ax=plt.subplots(figsize=(8,10))

sns.barplot(x='gbm_Col_Rank', y='Features', data=df_important_features.iloc[:30, :])

ax.set_title('Features in the order of importance for GBM Regressor', fontsize=12)

ax.set_xlabel('Rank', fontsize=12)

ax.set_ylabel('Columns/Features', fontsize=12)

plt.show()
rf_pred= model_rf.predict(X_val)

print('Random Forest Regressor accuracy score {}'.format(np.sqrt(mean_squared_log_error(y_val, rf_pred))))

lgb_pred= model_lgb.predict(X_val)

print('Light GBM accuracy score {}'.format(np.sqrt(mean_squared_log_error(y_val, rf_pred))))

xgb_pred= model_xgb.predict(X_val)

print('XG Boost Regressor accuracy score {}'.format(np.sqrt(mean_squared_log_error(y_val, rf_pred))))

gbm_pred= model_gbm.predict(X_val)

print('GBM Regressor accuracy score {}'.format(np.sqrt(mean_squared_log_error(y_val, rf_pred))))
from sklearn.model_selection import KFold

def kfold(rgr,X, Y):

    outcomes=[]

    kf=KFold(n_splits=5, random_state=0, shuffle=False)

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        X_dev, X_val= X.values[train_index], X.values[test_index]

        y_dev, y_val= Y[train_index], Y[test_index]

        rgr.fit(X_dev, y_dev)

        y_pred=rgr.predict(X_val)

        acc= np.sqrt(mean_squared_log_error(y_val, y_pred))

        outcomes.append(acc)

        print('Fold {0} accuracy {1:.2f}'.format(i,acc))

    mean_accuracy= np.mean(outcomes)

    return mean_accuracy
rf_pred= kfold(model_rf,X_train,Y_train)

print('Mean accuracy from Random Forest regression : {0:.4f}'.format(rf_pred))

'''lgb_pred= kfold(model_lgb,X_train,Y_train)

print('Mean accuracy from Light GBM regression : {0:.4f}'.format(lgb_pred))

xgb_pred= kfold(model_xgb,X_train,Y_train)

print('Mean accuracy from XG Boost regression : {0:.4f}'.format(xgb_pred))'''

gbm_pred= kfold(model_gbm,X_train,Y_train)

print('Mean accuracy from GBM regression : {0:.4f}'.format(gbm_pred))
def oof_pred(model, X, Y, X_test):

    Kfold=KFold(n_splits=5, random_state=0, shuffle=True)

    oof_train= np.zeros(X.shape[0])

    oof_test= np.zeros(df_test.shape[0])

    oof_test_kf=np.empty((Kfold.get_n_splits(), X_test.shape[0]))

    for i, (train_index, test_index) in enumerate(Kfold.split(X)):

        X_dev, X_val= X.values[train_index], X.values[test_index]

        y_dev, y_val= Y[train_index], Y[test_index]

        model.fit(X_dev, y_dev)

        

        oof_train[test_index]= model.predict(X_val)

        oof_test_kf[i,:]= model.predict(X_test)

        

    oof_test= oof_test_kf.mean(axis=0)

    return oof_train, oof_test                     
rf_pred_train, rf_pred_test= oof_pred(model_rf,X_train,Y_train,X_test)

'''lgb_pred_train, lgb_pred_test= oof_pred(model_lgb,X_train,Y_train,X_test)

xgb_pred_train, xgb_pred_test= oof_pred(model_xgb,X_train,Y_train,X_test)'''

gbm_pred_train, gbm_pred_test= oof_pred(model_gbm,X_train,Y_train,X_test)
train_final=pd.DataFrame({

    "Random Forest": rf_pred_train,

    "GBM": gbm_pred_train   

})
train_final.head()
fig, ax= plt.subplots(figsize=(8,8))

sns.heatmap(train_final.corr(),annot=True)

ax.set_title('Correlation between input features for the trained train data set', fontsize=12)

plt.show()
test_final=pd.DataFrame({

    "Random Forest":rf_pred_test,

    "GBM":gbm_pred_test

})
test_final.head()
fig, ax= plt.subplots(figsize=(8,8))

sns.heatmap(test_final.corr(),annot=True)

ax.set_title('Correlation between input features for the trained test data set', fontsize=12)

plt.show()
import xgboost as xgb

xgb_rgr=xgb.XGBRegressor(

n_estimators=2000,

max_depth=4,

min_child_weight=2,

gamma=0.9,

subsample=0.8,

colsample_bytree=0.8,

n_thread=-1,

scale_pos_weight=1)
train_final.shape,Y_train.shape,test_final.shape
sample_submission= pd.read_csv('../input/sample_submission.csv')
sample_submission.head()
xgb_rgr.fit(train_final, Y_train)

final_prediction= xgb_rgr.predict(test_final)

output= pd.DataFrame({

    "ID": sample_submission['ID'],

    "target": final_prediction    

})
output.head()