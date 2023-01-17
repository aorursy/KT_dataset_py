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
train=pd.read_csv('/kaggle/input/ieee-ml-hack/train.csv')
train
test=pd.read_csv('/kaggle/input/ieee-ml-hack/test.csv')
test
train.info()
import seaborn as sns
import matplotlib.pyplot as plt
cor_mat=train.corr().round(2)
plt.figure(figsize=(15,10))
sns.heatmap(data=cor_mat,annot=True)
#train['total']=train['50s']+train['6s']+train['Balls']
for i in train.columns:
    a=train[i].isnull().sum()
    if a>0:
        print('column {} with null value'.format(i), a)
train.drop(columns=['Name'],inplace=True)
test.drop(columns=['Name'],inplace=True)
#train["mean"]  = train.groupby(['ID'])['6s'].transform('mean')

from sklearn.model_selection import train_test_split
X = train.drop(columns=['Ratings','Innings','Maidens','Age','Balls','Economy_Rate','100s'])
Y = train['Ratings']
X_train, X_test, Y_train,Y_test = train_test_split(X,Y, test_size=0.3, random_state=7)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
#from sklearn.model_selection import cross_val_score
results = list()
strategies = ['mean', 'median', 'most_frequent', 'constant']
for s in strategies:

    pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestRegressor())])
   
    pipeline.fit(X_train,Y_train)
    scores = pipeline.score(X_test, Y_test)
    
    results.append(scores)
    print('%s %.3f'% (s, np.mean(scores)))
X_train = X_train.fillna(X_train.mean())
X_test=X_test.fillna(X_test.mean())
pipeline = Pipeline(steps=[('m', RandomForestRegressor())])
pipeline.fit(X_train, Y_train)
X_test.fillna(X_test.median())
scores = pipeline.score(X_test, Y_test)
print((scores))
from sklearn.metrics import mean_squared_error,r2_score
r_model = RandomForestRegressor(n_estimators=1800,max_features="auto",n_jobs=-1,max_samples=0.7,max_depth=10, random_state=0)
r_model.fit(X_train,Y_train)
y_pred = r_model.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(Y_test, y_pred)
print("R2 Score:", r2)
feature_important = r_model.feature_importances_
feature_important
total = sum(feature_important)
new = [value * 100. / total for value in feature_important]
new = np.round(new,2)
keys = list(X_train.columns)
feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new
feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
feature_importances

import xgboost as xgb
dtr =  xgb.XGBRegressor( max_depth=3,
                        min_child_weight=1,
                        gamma=5,
                     eta = 0.04,
            n_estimators = 1000 ,
                  subsample=0.8,
                        colsample_bytree=0.8,
                        seed=300)
dtr.fit(X_train,Y_train)
y_pred = dtr.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(Y_test, y_pred)
print("R2 Score:", r2)   
test_model = xgb.XGBRegressor(
            max_depth=3,
                        min_child_weight=1,
                        gamma=5,
                     eta = 0.04,
            n_estimators = 1000 ,
                  subsample=0.8,
                        colsample_bytree=0.8,
                        seed=300
)
#model.fit(X_train, y_train)
test_model.fit(X_train, Y_train, eval_metric='rmse', 
          eval_set=[(X_test, Y_test)], early_stopping_rounds=500, verbose=100)
hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l2', 'auc'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 15,
    'verbose': 0,
    "max_depth": 10,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 100000,
    "n_estimators": 1000
}
import lightgbm as lgb
gbm = lgb.LGBMRegressor(**hyper_params)

gbm.fit(X_train, Y_train,
        eval_set=[(X_test, Y_test)],
        eval_metric='l1',
        early_stopping_rounds=1000)

y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration_)
print('The r2_score of prediction is:', round(r2_score(y_pred, Y_train), 5))
from sklearn.neighbors import KNeighborsRegressor
dtr =  KNeighborsRegressor( n_neighbors=15, weights='distance', algorithm='brute', leaf_size=1000, p=1, metric='minkowski', n_jobs=-1)
dtr.fit(X_train,Y_train)
y_pred = dtr.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(Y_test, y_pred)
print("R2 Score:", r2)  
test
# g=test.iloc[:372]
# g
#test['Age']=test['Age'].map({'Un':42})
# test['Age']=test['Age'].map({'none':'42'})
# test['100s']=test['100s'].map({'none':'42'})
test['Balls']=test['Balls'].map({'none':'42'})
# test['Innings']=test['Innings'].map({'none':'42'})
# test['Economy_Rate']=test['Economy_Rate'].map({'none':'42'})
# test['Maidens']=test['Maidens'].map({'none':'42'})
test=test.fillna(test.mean())
test
# for i in test.columns:
#     a=test[i].isnull().sum()
#     if a>0:
#         print('column {} with null value'.format(i), a)
test.drop(columns=['Innings','Maidens','Age','Balls','Economy_Rate','100s'],inplace=True)
test_pred=gbm.predict(test)
test_pred
result=pd.DataFrame(test_pred)
result['Ratings']=pd.DataFrame(test_pred)
result['ID']=test['ID']
result.to_csv('a3.csv',index=False)
sub=pd.read_csv('/kaggle/input/ieee-ml-hack/sample_solution.csv')
sub