# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train_V2.csv")
df_test = pd.read_csv("../input/test_V2.csv")
df_train.info()
df_train.head()
df_train.isnull().sum()
df_test.shape
df_test.isnull().sum()
df_train[df_train.winPlacePerc.isnull()]
# fill null win PlacePerc by median
df_train.winPlacePerc.fillna(df_train.winPlacePerc.median(),inplace=True)
df_train.groupby("matchType")["winPlacePerc"].mean().sort_values().plot(kind = "bar")
df_train.shape
match_type = pd.get_dummies(df_train["matchType"])
df_train["playersInMatch"] = df_train.groupby("matchId")["Id"].transform("count")
df_train["playersInGroup"] = df_train.groupby("groupId")["Id"].transform("count")
df_train = pd.concat([df_train,match_type],sort=False, axis = 1)
df_train.drop(["matchType","matchId","groupId"],axis=1,inplace=True)
df_train.shape
corrTarget = df_train.corrwith(df_train.winPlacePerc).sort_values(ascending = False)
corrTarget.plot(kind="bar")
corr = df_train.corr()
import seaborn as sns
sns.heatmap(corr)
threshold = 0.9
high_corr_pair = {}
i = -1
for key,rows in corr.iterrows():
    i += 1
    for j,vals in enumerate(rows):
        if i != j:
            if abs(vals) > threshold:
                high_corr_pair["_".join([key,corr.columns[j]])] = vals
high_corr_pair
toRemoved = set()
for pair in high_corr_pair:
    a,b = pair.split("_")
    if corrTarget[a] <= corrTarget[b]:
        toRemoved.add(a)
    else:
        toRemoved.add(b)
    
toRemoved
df_train.drop(["killPoints","maxPlace","winPoints"], axis = 1, inplace = True)
sns.distplot(df_train["walkDistance"])
df_train["walkDistance"].quantile(0.99)
df_train["walkDistance"].min()
# clip walkDistance by 5000
#df_train.loc[df_train.walkDistance > 5000,"walkDistance"] = 5000
### have to check distribution
#import pandas_profiling as pdp
#pdp.ProfileReport(df_train)
match_type = pd.get_dummies(df_test["matchType"])
df_test["playersInMatch"] = df_test.groupby("matchId")["Id"].transform("count")
df_test["playersInGroup"] = df_test.groupby("groupId")["Id"].transform("count")
df_test = pd.concat([df_test,match_type],sort=False, axis = 1)
df_test.drop(["matchType","matchId","groupId"],axis=1,inplace=True)
df_test.drop(["killPoints","maxPlace","winPoints"], axis = 1, inplace = True)
df_train.set_index("Id",inplace=True)
import lightgbm as lgb #Boosting
from sklearn.ensemble import RandomForestRegressor #RF
from sklearn.linear_model import Ridge #Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df_train.drop(["winPlacePerc"], axis=1).values
Y = df_train["winPlacePerc"].values
X.shape
Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("X_train:",X_train.shape)
print("Y_train:",Y_train.shape)
print("X_test:",X_test.shape)
print("Y_test:",Y_test.shape)
#mini_train_size = 500000
#mini_test_size = 100000
#X_train_mini = X_train[:mini_train_size,:]
#Y_train_mini = Y_train[:mini_train_size]
#X_test_mini = X_test[:mini_test_size,:]
#Y_test_mini = Y_test[:mini_test_size]
#print("X_train_mini:",X_train_mini.shape)
#print("Y_train_mini:",Y_train_mini.shape)
#print("X_test_mini:",X_test_mini.shape)
#print("Y_test_mini:",Y_test_mini.shape)
#Linear: Ridge
#m_ridge = Ridge()
#m_ridge.fit(X_train,Y_train)
#m_ridge.score(X_test,Y_test)
#df_train.columns
#df_coef = pd.DataFrame({"coef":m_ridge.coef_,
#              "varName":[col for col in df_train.columns if col != "winPlacePerc" ]}).\
#                      set_index("varName")
#df_coef.sort_values(by = "coef").plot(kind="bar")
#Linear: Lasso
#from sklearn.linear_model import Lasso
#m_lasso = Lasso()
#m_lasso.fit(X_train,Y_train)
#m_lasso.score(X_test,Y_test)
#m_ridge = Ridge()
#m_ridge.fit(X_train_mini,Y_train_mini)
#m_ridge.score(X_test_mini,Y_test_mini)
#Linear: SGD
#from sklearn.linear_model import SGDRegressor
#m_sgd = SGDRegressor(max_iter = 500, verbose=2, early_stopping = True)
#m_sgd.fit(X_train_mini,Y_train_mini)
#m_sgd.score(X_test_mini,Y_test_mini)
%%time
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
lgbm_params = {'objective': 'regression','metric': 'rmse'}
m_lgb = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval, 
                  num_boost_round=2000,
                  early_stopping_rounds=5)
from sklearn.metrics import r2_score
Y_pred = m_lgb.predict(X_test, num_iteration=m_lgb.best_iteration)
r2_score(Y_test,Y_pred)
#%%time
#m_rf = RandomForestRegressor(n_estimators=100, n_jobs = -1,verbose=2)
#m_rf.fit(X_train_mini,Y_train_mini)
#print(m_rf.score(X_test_mini,Y_test_mini))
#df_importance_rf = pd.DataFrame({"importance":m_rf.feature_importances_,
#              "varName":[col for col in df_train.columns if col != "winPlacePerc" ]}).\
#                      set_index("varName")
#df_importance_rf.sort_values(by = "importance").plot(kind="bar")
#df_importance_rf_log = pd.DataFrame({"importance":np.log(m_rf.feature_importances_),
#              "varName":[col for col in df_train.columns if col != "winPlacePerc" ]}).\
#                      set_index("varName")
#df_importance_rf_log.sort_values(by = "importance").plot(kind="bar")
df_importance_lgb = pd.DataFrame({"importance":m_lgb.feature_importance(),
              "varName":[col for col in df_train.columns if col != "winPlacePerc" ]}).\
                      set_index("varName")
df_importance_lgb.sort_values(by = "importance").plot(kind="bar")
#lgb_train_full = lgb.Dataset(X_train, Y_train)
#lgb_eval_full = lgb.Dataset(X_test, Y_test, reference=lgb_train_full)
#lgbm_params = {'objective': 'regression','metric': 'rmse'}
#m_lgb_full = lgb.train(lgbm_params, lgb_train_full, valid_sets=lgb_eval_full, 
#                  num_boost_round=2000,
#                  early_stopping_rounds=5)
df_test.head()
X_for_pred = df_test.values[:,1:]
X_for_pred = sc.transform(X_for_pred)
Y_test_pred = m_lgb.predict(X_for_pred, num_iteration=m_lgb.best_iteration)
for_submission = pd.DataFrame({"Id":df_test.Id,"winPlacePerc":Y_test_pred})
for_submission.to_csv('submission.csv', index=False)
