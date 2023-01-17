import pandas as pd


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
test.head()
train.head()
train.info()
test.info()
sns.distplot(train['count']);
train["count"] = np.log1p(train["count"])
sns.distplot(train['count']);
ntrain = train.shape[0] # number of training
ntest = test.shape[0] #number of test

y_train = train["count"].values 
y_train
#concat train/test data
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.info()
all_data.shape
all_data.drop(['casual'], axis=1, inplace=True)
all_data.drop(['registered'], axis=1, inplace=True)
all_data.drop(['count'], axis=1, inplace=True)
all_data.shape
all_data.info()
#target = cout / 
#casual, registerd = 날려야?
import datetime
from datetime import datetime
all_data.head()
all_data['date']  = all_data.datetime.apply(lambda x: x.split()[0])
all_data['hour'] = all_data.datetime.apply(lambda x: x.split()[1].split(':')[0])
all_data['weekday'] = all_data.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())
all_data['month'] = all_data.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)
all_data.head()
all_data.drop(['datetime'], axis=1, inplace=True)
train = all_data[:ntrain]
test = all_data[ntrain:]
train["count"] = y_train
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import norm, skew 
#get the numeric values
numeric_features = all_data.dtypes[all_data.dtypes != "object"].index
numeric_features
# Check the skew of all numerical features
skewed_feats = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness
skewness = skewness[abs(skewness)>0.7]
skewness
#you can take log like this 

#skewness = skewness[abs(skewness)>0.7]
#all_data["holiday"] = np.log1p(all_data["holiday"])
#all_data["weather"] = np.log1p(all_data["weather"])
#all_data["workingday"] = np.log1p(all_data["workingday"])

#skewed_feats = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
#print("\nSkew in numerical features: \n")
#skewness = pd.DataFrame({'Skew' :skewed_feats})
#skewness
corr = train.corr(method='pearson').drop(['count']).sort_values('count', ascending=False)['count']
corr 
import seaborn as sns
corrMat = train.corr()
mask = np.array(corrMat)
mask[np.tril_indices_from(mask)] = False
fig, ax= plt.subplots(figsize=(20, 10))
sns.heatmap(corrMat, mask=mask,vmax=1., square=True,annot=True)
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
train.columns
train.head()
features = "atemp+holiday+humidity+season+temp+weather+windspeed+workingday+hour+weekday+month"
# Break into left and right hand side; y and X
y, X = dmatrices("count ~" + features, data=train, return_type="dataframe")

# For each Xi, calculate VIF
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Fit X to y
result = sm.OLS(y, X).fit()
print(result.summary())
# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
#이런식으로 drop해줄수 있다
#housing.drop('households', axis=1, inplace=True)
#housing.drop('latitude', axis=1, inplace=True)
train = all_data[:ntrain]
test = all_data[ntrain:]
train["count"]=y_train
group_season = train.groupby(['season'])['count'].sum().reset_index()
ax = sns.barplot(x = group_season['season'], y = group_season['count'])
ax.set(xlabel='season', ylabel='count')
plt.show()
group_dow = train.groupby(['weekday'])['count'].sum().reset_index()
ax = sns.barplot(x = group_dow['weekday'], y = group_dow['count'])
ax.set(xlabel='weekday', ylabel='count')
plt.show()
group_mn = train.groupby(['month'])['count'].sum().reset_index()
ax = sns.barplot(x = group_mn['month'], y = group_mn['count'])
ax.set(xlabel='month', ylabel='count')
plt.show()
group_hr = train.groupby(['hour'])['count'].sum().reset_index()
ax = sns.barplot(x = group_hr['hour'], y = group_hr['count'])
ax.set(xlabel='hour', ylabel='count')
plt.show()
plt.figure(figsize=(29,15))
group_season = train.groupby(['temp'])['count'].sum().reset_index()
ax = sns.barplot(x = group_season['temp'], y = group_season['count'])
ax.set(xlabel='temp', ylabel='count')
plt.show()
plt.figure(figsize=(30,12))
group_season = train.groupby(['atemp'])['count'].sum().reset_index()
ax = sns.barplot(x = group_season['atemp'], y = group_season['count'])
ax.set(xlabel='atemp', ylabel='count')
plt.show()
all_data["temp"].value_counts()
train_test_data = [train, test] # train과 test set 합침
for dataset in train_test_data:
    dataset.loc[ dataset['temp'] <= 5, 'temp'] = 0,
    dataset.loc[(dataset['temp'] > 5) & (dataset['temp'] <= 10), 'temp'] = 1,
    dataset.loc[(dataset['temp'] > 10) & (dataset['temp'] <= 15), 'temp'] = 2,
    dataset.loc[(dataset['temp'] > 15) & (dataset['temp'] <= 20), 'temp'] = 3,
    dataset.loc[(dataset['temp'] > 20) & (dataset['temp'] <= 25), 'temp'] = 4,
    dataset.loc[(dataset['temp'] > 25) & (dataset['temp'] <= 30), 'temp'] = 5,        
    dataset.loc[(dataset['temp'] > 30) & (dataset['temp'] <= 35), 'temp'] = 6, 
    dataset.loc[ dataset['temp'] > 35, 'temp'] = 7
train.head()
plt.figure(figsize=(12,7))
group_season = train.groupby(['temp'])['count'].sum().reset_index()
ax = sns.barplot(x = group_season['temp'], y = group_season['count'])
ax.set(xlabel='temp', ylabel='count')
plt.show()
plt.figure(figsize=(29,15))
group_season = train.groupby(['humidity'])['count'].sum().reset_index()
ax = sns.barplot(x = group_season['humidity'], y = group_season['count'])
ax.set(xlabel='humidity', ylabel='count')
plt.show()
for dataset in train_test_data:
    dataset.loc[ dataset['humidity'] <= 10, 'humidity'] = 0,
    dataset.loc[(dataset['humidity'] > 10) & (dataset['humidity'] <= 20), 'humidity'] = 1,
    dataset.loc[(dataset['humidity'] > 20) & (dataset['humidity'] <= 30), 'humidity'] = 2,
    dataset.loc[(dataset['humidity'] > 30) & (dataset['humidity'] <= 40), 'humidity'] = 3,
    dataset.loc[(dataset['humidity'] > 40) & (dataset['humidity'] <= 50), 'humidity'] = 4,
    dataset.loc[(dataset['humidity'] > 50) & (dataset['humidity'] <= 60), 'humidity'] = 5,        
    dataset.loc[(dataset['humidity'] > 60) & (dataset['humidity'] <= 70), 'humidity'] = 6, 
    dataset.loc[(dataset['humidity'] > 70) & (dataset['humidity'] <= 80), 'humidity'] = 7, 
    dataset.loc[(dataset['humidity'] > 80) & (dataset['humidity'] <= 90), 'humidity'] = 8,  
    dataset.loc[ dataset['humidity'] > 90, 'humidity'] = 9
 
plt.figure(figsize=(12,7))
group_season = train.groupby(['temp'])['count'].sum().reset_index()
ax = sns.barplot(x = group_season['temp'], y = group_season['count'])
ax.set(xlabel='temp', ylabel='count')
plt.show()
all_data.info()
train.head()
test.head()
train.drop('atemp', axis=1, inplace=True)
train.drop('date', axis=1, inplace=True)
test.drop('atemp', axis=1, inplace=True)
test.drop('date', axis=1, inplace=True)
train.head()
train.drop('count', axis=1, inplace=True)
#concat train/test data
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.head()
all_data.info()
#Changing into a categorical variable
all_data['season'] = all_data['season'].astype(str)
all_data['weather'] = all_data['weather'].astype(str)
all_data['hour'] = all_data['hour'].astype(str)
all_data['weekday'] = all_data['weekday'].astype(str)
all_data['month'] = all_data['month'].astype(str)
all_data['workingday'] = all_data['workingday'].astype(str)
all_data['holiday'] = all_data['holiday'].astype(str)


all_data.info()
#from sklearn.preprocessing import LabelEncoder
#cols = ('season', 'weather', 'hour', 'weekday', 'month', 'workingday','holiday')
# process columns, apply LabelEncoder to categorical features
#for c in cols:
#    lbl = LabelEncoder() 
#    lbl.fit(list(all_data[c].values)) 
#    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
#all_data.shape
all_data = pd.get_dummies(all_data)
print(all_data.shape)
all_data.head()
train = all_data[:ntrain]
test = all_data[ntrain:]
train.head()
test.head()
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))
lModel = LinearRegression()

# Train the model
lModel.fit(X = train,y = y_train)
lModel.fit(train,y_train)
lModel_train_pred = lModel.predict(train)
lModel_pred = np.expm1(lModel.predict(test.values))
print ("RMSLE Value For Linear Regression: ")
print(rmsle(y_train, lModel_train_pred))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=3))
lasso.fit(train,y_train)
lasso_train_pred = lasso.predict(train)
lasso_pred = np.expm1(lasso.predict(test.values))
print ("RMSLE Value For Lasso Regression: ")
print(rmsle(y_train, lasso_train_pred))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
ENet.fit(train,y_train)
ENet_train_pred = ENet.predict(train)
ENet_pred = np.expm1(ENet.predict(test.values))
print ("RMSLE Value For ENet Regression: ")
print(rmsle(y_train, ENet_train_pred))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5) #kernel = 'rbf' , 'sigmoid' 
KRR.fit(train,y_train)
KRR_train_pred = KRR.predict(train)
KRR_pred = np.expm1(KRR.predict(test.values))
print ("RMSLE Value For KRR Regression: ")
print(rmsle(y_train, KRR_train_pred))
GBoost = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='huber', max_depth=4,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=15, min_samples_split=10,
             min_weight_fraction_leaf=0.0, n_estimators=3000,
             presort='auto', random_state=5, subsample=1.0, verbose=0,
             warm_start=False)
GBoost.fit(train,y_train)
GBoost_train_pred = GBoost.predict(train)
GBoost_pred = np.expm1(GBoost.predict(test.values))
print ("RMSLE Value For GBoost Regression: ")
print(rmsle(y_train, GBoost_train_pred))
#You can do grid search like this 

#scorer = make_scorer(rmsle, greater_is_better=False)

#params = {'n_estimators':[3000, 3200, 3500], 'learning_rate' :[0.1, 0.05]}


#model_gb = GradientBoostingRegressor(n_estimators=3200, learning_rate=0.1,
#                                   max_depth=4, max_features='sqrt',
#                                   min_samples_leaf=15, min_samples_split=10, 
#                                   loss='huber', random_state =5)

#grid_search = GridSearchCV(model_gb, params, cv=5, scoring=scorer)
#grid_search.fit(train, y_train)

#grid_search.best_estimator_
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
model_svr = SVR(C=1, cache_size=200, coef0=0, degree=3, epsilon=0.1, gamma='auto',
  kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

#‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
model_svr.fit(train, y_train)
svr_train_pred = model_svr.predict(train)
svr_pred = np.expm1(model_svr.predict(test.values))
#You can do grid search like this print(rmsle(y_train, svr_train_pred))
#You can do grid search like this 

#params = {'coef0':[0, 0.1, 0.5, 1], 'C' :[0.1, 0.5, 1], 'epsilon':[0.1, 0.3, 0.5]}


#model_svr = SVR()
#grid_search = GridSearchCV(model_svr, params, cv=3, scoring=scorer)
#grid_search.fit(train, y_train)
#grid_search.best_estimator_
regr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=30, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=60, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
regr.fit(train, y_train)
regr_train_pred = regr.predict(train)
regr_pred = np.expm1(regr.predict(test.values))
print(rmsle(y_train, regr_train_pred))
#You can do grid search like this 

#param_grid = [
#    {'n_estimators': [3, 10, 30, 60, 90], 'max_features': [10,20,30,40,50]},
#    {'bootstrap': [True], 'n_estimators': [3, 10, 30, 60, 90], 'max_features': [10,20,30,40,50]},
#]

#forest_reg = RandomForestRegressor()
#grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring=scorer)
#grid_search.fit(train, y_train)
#grid_search.best_estimator_
#ensemble = xgb_pred*0.25 + GBoost_pred*0.25 + regr_pred*0.5  
#ensemble = xgb_pred*0.3 + GBoost_pred*0.3 + regr_pred*0.4  
ensemble = xgb_pred*0.4 + GBoost_pred*0.4 + regr_pred*0.2  # 이게젤높다  this one gives me a best score 
#test = pd.read_csv('../input/test.csv')
#timeColumn = test['datetime']
#sub = pd.DataFrame()
#sub['datetime'] = timeColumn
#sub['count'] = ensemble
#sub.to_csv('submission.csv',index=False)
