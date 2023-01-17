import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape
train_ids = train.Id
test_ids = test.Id
train.drop('Id', inplace=True, axis=1)
test.drop('Id', inplace=True, axis=1)
target = train.SalePrice
train.drop('SalePrice', axis=1, inplace=True)
for col in train.columns:
    if (train[col].dtype=='object' and train[col].isnull().sum() > 0):
        filling_value = train[col].value_counts().index[0]
        train[col].fillna(filling_value, axis=0, inplace=True)        
    else:
        if train[col].isnull().sum()>0:
            filling_value = train[col].median()
            train[col].fillna(filling_value, axis=0, inplace=True)
for col in test.columns:
    if (test[col].dtype=='object' and test[col].isnull().sum() > 0):
        filling_value = test[col].value_counts().index[0]
        test[col].fillna(filling_value, axis=0, inplace=True)        
    else:
        if test[col].isnull().sum()>0:
            filling_value = test[col].median()
            test[col].fillna(filling_value, axis=0, inplace=True)
all_data = pd.concat([train, test], sort=False)
all_data = pd.get_dummies(all_data)
train_ = all_data[:train.shape[0]]
test_ = all_data[train.shape[0]:]

train_['SalePrice'] = target.values
print(train_.shape)
print(test_.shape)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
isof = IsolationForest(n_estimators=1000, random_state=4747, n_jobs=-1, behaviour='auto')
isof.fit(train_)

isof_predict = isof.predict(train_)
isof_predict = pd.DataFrame(isof_predict, columns=['Value'])
train_ready = train_.iloc[isof_predict[ isof_predict.Value==1 ].index.values]
train_ids = train_ids.iloc[isof_predict[isof_predict.Value==1].index.values]
#train_ready.reset_index(inplace=True, drop=True)
print('Outlier: ', isof_predict[isof_predict.Value!=1].shape[0])
print(train_.shape)
print(train_ready.shape)
"""
isof = IsolationForest(n_estimators=1000, random_state=4747, n_jobs=-1, behaviour='auto')
isof.fit(test_)

isof_predict = isof.predict(test_)
isof_predict = pd.DataFrame(isof_predict, columns=['Value'])
test_ready = test_.iloc[isof_predict[ isof_predict.Value==1 ].index.values]
test_ids = test_ids.iloc[isof_predict[isof_predict.Value==1].index.values]
#test_ready.reset_index(inplace=True, drop=True)
print('Outlier: ', isof_predict[isof_predict.Value!=1].shape[0])
print(test_.shape)
print(test_ready.shape)
"""
#train_model = train_.drop('SalePrice', axis=1, inplace=False)
#target_model = train_['SalePrice']

train_model = train_ready.drop('SalePrice', axis=1, inplace=False)
target_model = train_ready['SalePrice']
test_model = test_
import seaborn as sns
from scipy import stats

sns.distplot(target_model, fit=stats.norm)

(mu, sigma) = stats.norm.fit(target_model)

print('\n mu= {:.2f} and sigma= {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(target_model, plot=plt)
plt.show()
target_log_trans = np.log( target_model )
sns.distplot( target_log_trans , fit=stats.norm)

(mu, sigma) = stats.norm.fit(target_log_trans)

print('\n mu= {:.2f} and sigma= {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(target_log_trans, plot=plt)
plt.show()
"""
from sklearn.decomposition import PCA

_pca = PCA(n_components=train_model.shape[1])

train_model = _pca.fit_transform(train_model)
test_model = _pca.fit_transform(test_model)
"""
#tried but there is no improvment. also decreased the r2 score. 
print(train_.shape)
print(train_model.shape)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import model_selection

from xgboost import XGBRegressor
import xgboost as xgb

xgb_model = XGBRegressor(n_jobs=-1, max_depth=3, min_child_weight=3,n_estimators=999, subsample=0.9)

train_x, test_x, train_y, test_y = train_test_split(train_model,np.log(target_model), test_size=19, 
                                                    random_state=47)

xgb_model.fit(X=train_x, y= train_y)

preds = xgb_model.predict(test_x)
print("r2: {:.4f}".format(r2_score(test_y, preds) ) )
print("mse: {:.4f}".format(mean_squared_error(test_y, preds) ) )
plt.figure(figsize=(40,20))
xgb.plot_importance(xgb_model)
"""
from sklearn.feature_selection import SelectFromModel

thresholds = sorted(xgb_model.feature_importances_, reverse=True)
importances = pd.DataFrame()

for thresh in thresholds:
    selection = SelectFromModel(xgb_model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(train_x)
    selection_model = XGBRegressor()
    selection_model.fit(select_X_train, train_y)
    
    select_X_test = selection.transform(test_x)
    selection_preds = selection_model.predict(select_X_test)
    r2 = r2_score(test_y, selection_preds)
    #print("Thres= %.3f, n=%d, r2:%.4f" % (thresh, select_X_train.shape[1], r2))
    sub_df = pd.DataFrame({'thresh':"{:.3f}".format(thresh), 'n':"{:d}".format(select_X_train.shape[1]), 'r2':"{:.4f}".format(r2)}, index=[0])
    #print(sub_df)
    importances = importances.append(sub_df)
    
importances.reset_index(inplace=True)
importances.head()    
"""

#looked at features but there is no improvment. 
from lightgbm import LGBMRegressor
import lightgbm as lgb

lgb_model = LGBMRegressor(n_jobs=-1)

train_x, test_x, train_y, test_y = train_test_split(train_model,np.log(target_model), test_size=19, 
                                                    random_state=47)

lgb_model.fit(train_x, train_y)

preds_lgbm = lgb_model.predict(test_x)

print("r2: {:.4f}".format(r2_score(test_y, preds_lgbm) ) )
print("mse: {:.4f}".format(mean_squared_error(test_y, preds_lgbm) ) )
lgb.plot_importance(lgb_model,figsize=(40,20))
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor, GradientBoostingRegressor

ab = AdaBoostRegressor(random_state=4747)
bg = BaggingRegressor(max_features=0.33, n_jobs=-1, random_state=4747)
gbr = GradientBoostingRegressor(random_state=4747)
train_x, test_x, train_y, test_y = train_test_split(train_model,np.log(target_model), test_size=19, 
                                                    random_state=47)

ab.fit(train_x, train_y)
bg.fit(train_x, train_y)
gbr.fit(train_x, train_y)

pred_ab = ab.predict(test_x)
pred_bg = bg.predict(test_x)
pred_gbr=gbr.predict(test_x)

print('AdaBoost Regression')
print("r2: {:.4f}".format(r2_score(test_y, pred_ab) ) )
print("mse: {:.4f}".format(mean_squared_error(test_y, pred_ab) ) )

print('Bagging Regression')
print("r2: {:.4f}".format(r2_score(test_y, pred_bg) ) )
print("mse: {:.4f}".format(mean_squared_error(test_y, pred_bg) ) )

print('GradientBoosting Regression')
print("r2: {:.4f}".format(r2_score(test_y, pred_gbr) ) )
print("mse: {:.4f}".format(mean_squared_error(test_y, pred_gbr) ) )

[x for x in (range(3, 10, 1))]
from sklearn.model_selection import GridSearchCV

#gbr parameter tuning
params_grid = {
    'loss':['ls','lad'],
    'learning_rate':[0.1, 0.5, 0.01, 0.05],
    'n_estimators': [x for x in range(100, 1100, 100)],
    'max_depth':[x for x in (range(3, 10, 1))]
}

estimator = GradientBoostingRegressor(random_state=4747)

grid_src = GridSearchCV(estimator=estimator, param_grid=params_grid, cv=5, n_jobs=-1, verbose=2)

grid_src.fit(train_x, train_y)
    




submission_preds = lgb_model.predict(test_model)
submission_preds_transformed = np.exp(submission_preds)

df_lgb_submission = pd.DataFrame({'Id':test_ids, 'SalePrice':np.ceil(submission_preds_transformed)})
df_lgb_submission.head()
submission_preds = xgb_model.predict(test_model)
submission_preds_transformed = np.exp(submission_preds)

df_xgb_submission = pd.DataFrame({'Id':test_ids, 'SalePrice':np.ceil(submission_preds_transformed)})
df_xgb_submission.head()
merge_df = pd.merge(df_lgb_submission,df_xgb_submission, how='inner', left_on='Id', right_on='Id')
merge_df.head()
merge_df['SalePrice'] = (merge_df.SalePrice_x + merge_df.SalePrice_y) / 2
merge_df.drop(['SalePrice_x','SalePrice_y'], axis=1, inplace=True)
merge_df.head()
merge_df.to_csv('submission.csv', index=False)
