import warnings

warnings.filterwarnings("ignore")

#Importing necessary Libraries

%matplotlib inline

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,Ridge

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb

from mlxtend.regressor import StackingRegressor,StackingCVRegressor
# Columns of the Dataset

datadict=pd.read_excel('../input/Data_Dictionary.xlsx')

datadict
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head(3)
train.info()
# Checking for travellers who have previos bookings

res_id=pd.Series(train['memberid'].tolist())

booking_more_than_one=np.sum(res_id.value_counts()>1)

booking_more_than_one
train.isnull().sum()
temp_season_holidayed_code=train[train['season_holidayed_code'].notna()]

temp_state_code_residence=train[train['state_code_residence'].notna()]

train['season_holidayed_code']=train['season_holidayed_code'].fillna(temp_season_holidayed_code['season_holidayed_code'].mode()[0])

train['state_code_residence']=train['state_code_residence'].fillna(temp_season_holidayed_code['state_code_residence'].mode()[0])

test['season_holidayed_code']=test['season_holidayed_code'].fillna(temp_season_holidayed_code['season_holidayed_code'].mode()[0])

test['state_code_residence']=test['state_code_residence'].fillna(temp_season_holidayed_code['state_code_residence'].mode()[0])

sns.distplot(train['amount_spent_per_room_night_scaled'])
print('Skewness :',train['amount_spent_per_room_night_scaled'].skew())

print('Kurtosis :',train['amount_spent_per_room_night_scaled'].kurt())
sns.set(style='whitegrid')

sns.boxplot(y='amount_spent_per_room_night_scaled',x='resort_type_code',data=train)


sns.boxplot(y='amount_spent_per_room_night_scaled',x='channel_code',data=train)


sns.boxplot(y='amount_spent_per_room_night_scaled',x='room_type_booked_code',data=train)
corrmat=train.corr(method='spearman')

f,ax=plt.subplots(figsize=(12,6))

sns.heatmap(corrmat,ax=ax,cmap="YlGnBu", linewidths=0.1)
cols=['numberofadults','total_pax','amount_spent_per_room_night_scaled']

sns.pairplot(train[cols],size=4.5)
f,ax=plt.subplots(figsize=(15,5))

sns.boxplot(y='amount_spent_per_room_night_scaled',x='total_pax',data=train,ax=ax)
import collections

mem_id=pd.Series(train['memberid'].tolist()).astype(str)

prev_bookings=[]

ctr=collections.Counter(mem_id)

for ele in mem_id:

    prev_bookings.append(ctr.get(ele))

prev_bookings=[(lambda x:x-1)(x)for x in prev_bookings]
train['prev_bookings']=prev_bookings
f,ax=plt.subplots(figsize=(15,5))

sns.boxplot(y='amount_spent_per_room_night_scaled',x='prev_bookings',data=train,ax=ax)
plt.figure(figsize=(10, 5))

sns.distplot(train[train['numberofadults'] >3 ]['amount_spent_per_room_night_scaled'][0:] , label = "1", color = 'red')

sns.distplot(train[train['numberofadults'] ==0 ]['amount_spent_per_room_night_scaled'][0:] , label = "1", color = 'green')

plt.show()
plt.figure(figsize=(10, 5))

sns.distplot(train[train['prev_bookings'] >15 ]['amount_spent_per_room_night_scaled'][0:] , label = "1", color = 'green')

sns.distplot(train[train['prev_bookings'] == 0]['amount_spent_per_room_night_scaled'][0:] , label = "0" , color = 'red' )

plt.show()


train.plot.scatter(x='total_pax',y='amount_spent_per_room_night_scaled')


train.plot.scatter(x='roomnights',y='amount_spent_per_room_night_scaled')
temp=train[train['roomnights']<0]

temp
train=train[train['roomnights']>0]

train.plot.scatter(x='roomnights',y='amount_spent_per_room_night_scaled')
train.columns
# Preparing Train Data

cat_cols=['channel_code','main_product_code','resort_region_code','resort_type_code','room_type_booked_code','state_code_residence'

         ,'state_code_resort','member_age_buckets','booking_type_code','reservationstatusid_code','cluster_code']

for c in cat_cols:

    lb=LabelEncoder()

    lb.fit(list(train[c].values))

    train[c]=lb.transform(list(train[c].values))

    
col_drop=['reservation_id','booking_date','checkin_date','checkout_date','amount_spent_per_room_night_scaled','memberid','resort_id']

y_train=train['amount_spent_per_room_night_scaled']

x_train=train.drop(col_drop,axis=1)
print(x_train.shape)

print(y_train.shape)
# Preparing Test Data

cat_cols=['channel_code','main_product_code','resort_region_code','resort_type_code','room_type_booked_code','state_code_residence'

         ,'state_code_resort','member_age_buckets','booking_type_code','reservationstatusid_code','cluster_code']

col_drop=['reservation_id','booking_date','checkin_date','checkout_date','memberid','resort_id']

for c in cat_cols:

    lb=LabelEncoder()

    lb.fit(list(test[c].values.astype(str)))

    test[c]=lb.transform(list(test[c].values.astype(str)))

x_test=test.drop(col_drop,axis=1)

mem_id=pd.Series(test['memberid'].tolist()).astype(str)

prev_bookings=[]

ctr=collections.Counter(mem_id)

for ele in mem_id:

    prev_bookings.append(ctr.get(ele))

prev_bookings=[(lambda x:x-1)(x)for x in prev_bookings]

x_test['prev_bookings']=prev_bookings

print(x_test.shape)
def cross_val(model):

    kf=KFold(n_splits=5,shuffle=True).get_n_splits(x_train)

    rmse=100*np.sqrt(-cross_val_score(model,x_train,y_train,scoring='neg_mean_squared_error',cv=kf,n_jobs=-1))

    return rmse
from sklearn.model_selection import RandomizedSearchCV

params={'n_estimators':[500,1000,2000,3000],'learning_rate':[0.008,0.01,0.05]}

def randomsearch_xgb():

    model=xgb.XGBRegressor()

    clf=RandomizedSearchCV(model,params,n_jobs=-1,scoring='neg_mean_squared_error')

    clf.fit(x_train,y_train)

    cv_data=pd.DataFrame(clf.cv_results_)

    return cv_data

    

    

    
# function to plot result of hyperparameter tunning using gridsearchCV

def plotResult(data):

    max_scores=data.groupby(['param_n_estimators','param_learning_rate']).max()

    max_scores=max_scores.unstack()[['mean_test_score','mean_train_score']]

    sns.heatmap((-1)*max_scores.mean_test_score,annot=True,fmt='.4g')
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ridge = make_pipeline(RobustScaler(), Ridge(alpha =0.0005, random_state=4))
enet=make_pipeline(RobustScaler(),ElasticNet(alpha =0.0005, random_state=1))
gbr=GradientBoostingRegressor(loss='huber',n_estimators=2000,min_samples_split=15,max_depth=4,learning_rate=0.05,verbose=1,n_iter_no_change=10)
rf=RandomForestRegressor(n_estimators=500,min_samples_split=15,max_depth=3)
# Cross validation using Randomized Search

cv_data=randomsearch_xgb()

plotResult(cv_data)
plotResult(cv_data)
xgbr = xgb.XGBRegressor(gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2300,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1,early_stopping_rounds=5)
lgbmr=lgb.LGBMRegressor(learning_rate=0.05, n_estimators=2500)

scores=cross_val(lasso)

print('Lasso Regressor RMSE: {},Std. Dev:{} :'.format(scores.mean(),scores.std()))
scores=cross_val(enet)

print('Elastic Net Regressor RMSE: {},Std. Dev:{} :'.format(scores.mean(),scores.std()))
scores=cross_val(ridge)

print('Ridge Regressor RMSE: {},Std. Dev:{} :'.format(scores.mean(),scores.std()))
scores=cross_val(xgbr)

print('XGBoost Regressor RMSE: {},Std. Dev:{} :'.format(scores.mean(),scores.std()))
scores=cross_val(lgbmr)

print('LightGBM Regressor RMSE: {},Std. Dev:{} :'.format(scores.mean(),scores.std()))
scores=cross_val(rf)

print('RandomForest Regressor RMSE: {},Std. Dev:{} :'.format(scores.mean(),scores.std()))
alpha=[0.0001,0.0005,0.0010,0.0050,0.01,0.05]

for i in alpha:

    lasso = make_pipeline(RobustScaler(), Lasso(alpha =i, random_state=1))

    k=KFold(n_splits=5,shuffle=True).get_n_splits(x_train)

    rmse=100*np.sqrt(-cross_val_score(lasso,x_train,y_train,scoring='neg_mean_squared_error',cv=k,n_jobs=-1))

    print('RMSE : {} ,STDEV:{} for alpha:{}'.format(rmse.mean(),rmse.std(),i))
# Final Lasso model

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.001, random_state=1))
# Tunning for n_estimators

n_est=[500,1000,1500,2000,2500,3000]

for n in n_est:

    lgbm=lgb.LGBMRegressor(learning_rate=0.05,min_child_samples=50, n_estimators=n,

                               bagging_fraction = 0.8)

    

    kf=KFold(n_splits=5,shuffle=True).get_n_splits(x_train)

    rmse=100*np.sqrt(-cross_val_score(lgbm,x_train,y_train,scoring='neg_mean_squared_error',cv=kf,n_jobs=-1))

    print('RMSE : {} ,STDEV:{} for n_estimators:{}'.format(rmse.mean(),rmse.std(),n))
# Tunning for min samples

min_samples=[10,20,30,40,50,100,200]

for n in min_samples:

    lgbm=lgb.LGBMRegressor(learning_rate=0.05,min_child_samples=n, n_estimators=1500,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9)

    

    kf=KFold(n_splits=5,shuffle=True).get_n_splits(x_train)

    rmse=100*np.sqrt(-cross_val_score(lgbm,x_train,y_train,scoring='neg_mean_squared_error',cv=kf,n_jobs=-1))

    print('RMSE : {} ,STDEV:{} for min_child_samples:{}'.format(rmse.mean(),rmse.std(),n))
#Tunning for Learning rate

learning_rate=[0.005,0.01,0.03,0.05]

for n in learning_rate:

    lgbm=lgb.LGBMRegressor(learning_rate=n,min_child_samples=100, n_estimators=1500,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9)

    

    kf=KFold(n_splits=5,shuffle=True).get_n_splits(x_train)

    rmse=100*np.sqrt(-cross_val_score(lgbm,x_train,y_train,scoring='neg_mean_squared_error',cv=kf,n_jobs=-1))

    print('RMSE : {} ,STDEV:{} for Learnig Rate:{}'.format(rmse.mean(),rmse.std(),n))
# Final Tuned LightGBM Model

lgbm=lgb.LGBMRegressor(learning_rate=0.03,min_child_samples=100, n_estimators=1500,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9)
#https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X.iloc[train_index], y.iloc[train_index])

                y_pred = instance.predict(X.iloc[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (enet, xgbr, rf),

                                                 meta_model = lgbm)

score = cross_val(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
stacked_averaged_models.fit(x_train,y_train)

lgbmr.fit(x_train,y_train)

xgbr.fit(x_train,y_train)

y_pred_lgbm=lgbmr.predict(x_test)

y_pred_xgb=xgbr.predict(x_test)

y_pred_strgr=stacked_averaged_models.predict(x_test)

y_pred_final=y_pred_lgbm*0.15+y_pred_xgb*0.15+y_pred_strgr*0.80


sub=pd.DataFrame()

sub['reservation_id']=test['reservation_id']

sub['amount_spent_per_room_night_scaled']=y_pred_final
sub.to_csv('Submission.csv',index=False)