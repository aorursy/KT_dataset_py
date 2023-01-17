import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pandas_profiling
from  matplotlib.gridspec import GridSpec

%matplotlib inline
import warnings

warnings.filterwarnings('ignore')
bike_train = pd.read_csv('../input/bike-sharing-demand/train.csv')
bike_test = pd.read_csv('../input/bike-sharing-demand/test.csv')
bike_train.shape
datetime = bike_test['datetime']
bike_train.head()
bike_train.columns.unique()
# bike_train.isnull().sum()
msno.matrix(bike_train)
msno.matrix(bike_test)
# profile = bike_train.profile_report(title = 'Pandas Profile Report')
# profile.to_file(output_file = 'Bike Sharing profile.html')
sns.set(style='whitegrid',color_codes=True)
# from matplotlib import style
# style.use('fivethirtyeight')
sns.boxplot(data=bike_train[['datetime', 'temp','atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])
fig = plt.gcf()            # 获取 fig 对象
fig.set_size_inches(10,10) # 设置图像的长度和宽度
for col in ['casual','registered','count']:
    bike_train['{}_log'.format(col)] = np.log1p(bike_train[col])
bike_train.head(2)
plt.figure(figsize=(15,15))
plt.subplot(321)
sns.distplot(bike_train['casual'])
plt.xlabel("casual (before transformation)")
plt.subplot(322)
sns.distplot(bike_train['casual_log'])
plt.xlabel("casual_log (after transformation)")
plt.subplot(323)
sns.distplot(bike_train['registered'])
plt.xlabel("registered (before transformation)")
plt.subplot(324)
sns.distplot(bike_train['registered_log'])
plt.xlabel("registered_log (after transformation)")
plt.subplot(325)
sns.distplot(bike_train['count'])
plt.xlabel("count (before transformation)")
plt.subplot(326)
sns.distplot(bike_train['count_log'])
plt.xlabel("count_log (after transformation)")
pd.DatetimeIndex(bike_train['datetime'])
bike_train['hour']  = [ t.hour for t in pd.DatetimeIndex(bike_train['datetime'])]
bike_train['dayofweek']  = [ t.dayofweek for t in pd.DatetimeIndex(bike_train['datetime'])]
bike_train['month']  = [ t.month for t in pd.DatetimeIndex(bike_train['datetime'])]
bike_train['year']  = [ t.year for t in pd.DatetimeIndex(bike_train['datetime'])]
# bike_train['year'] = bike_train['year'].map({2011:0,2012:1})
bike_train.drop(columns='datetime',inplace=True)
bike_train.head()
bike_test['hour']  = [ t.hour for t in pd.DatetimeIndex(bike_test['datetime'])]
bike_test['dayofweek']  = [ t.dayofweek  for t in pd.DatetimeIndex(bike_test['datetime'])]
bike_test['month']  = [ t.month for t in pd.DatetimeIndex(bike_test['datetime'])]
bike_test['year']  = [ t.year for t in pd.DatetimeIndex(bike_test['datetime'])]
bike_test.drop(columns='datetime',inplace=True)
bike_test.head()
#### new_feature  year + season
bike_train['year_season'] = bike_train['year'] +  bike_train['season']/10
bike_test['year_season'] = bike_test['year'] +  bike_test['season']/10
fig = plt.figure(figsize=(15,12))
gls = GridSpec(4,4,fig,wspace=0.5,hspace=0.5)
plt.subplot(gls[:2,:])
sns.boxplot(x='year_season',y='count',data=bike_train)
plt.subplot(gls[2:,:2])
sns.boxplot(x='year_season',y='casual',data=bike_train)
plt.subplot(gls[2:,2:])
sns.boxplot(x='year_season',y='registered',data=bike_train)
fig = plt.figure(figsize=(15,12))
gls = GridSpec(4,4,fig,wspace=0.5,hspace=0.5)
plt.subplot(gls[:2,:])
sns.boxplot(x='hour',y='count',hue='workingday',data=bike_train)
plt.subplot(gls[2:,:2])
sns.boxplot(x='hour',y='casual',hue='workingday',data=bike_train)
plt.subplot(gls[2:,2:])
sns.boxplot(x='hour',y='registered',hue='workingday',data=bike_train)
# new_feature casual/registered + hour + workingday 之后要预测 casual 和 registered 的值，这里构造出和它相关的特征。
# 在三个联合属性中 将casual和registered按照高低进行标记（0/1）
for df in [bike_train,bike_test]:
    df['hour_workingday_casual'] = df[['hour','workingday']].apply(lambda x:int(9 <= x['hour'] <= 20),axis=1)
    df['hour_workingday_registered'] = df[['hour','workingday']].apply(lambda x:int(
        (x['workingday']== 1 and (x['hour'] == 8 or 17 <= x['hour'] <= 18)) or (x['workingday']== 0 and (x['hour'] == 8 or 10 <= x['hour'] <= 19))
                            ),axis=1)
    
by_season = bike_train.groupby(['year_season'])[['count']].median()
by_season.columns = ['count_season']
bike_train = bike_train.join(by_season, on='year_season')
bike_test = bike_test.join(by_season, on='year_season')
from sklearn.ensemble import RandomForestRegressor
# RMSLE Score 0.38532
casual_features=['season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed',
            'year', 'hour', 'dayofweek', 'hour_workingday_casual', 'count_season']

rdr = RandomForestRegressor(n_estimators=500,random_state=10)
rdr.fit(bike_train[casual_features],bike_train['casual_log'])
pred_casual = rdr.predict(bike_test[casual_features])
pred_casual = np.expm1(pred_casual)
pred_casual[pred_casual < 0 ] = 0

registered_features=['season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed',
            'year', 'hour', 'dayofweek', 'hour_workingday_registered', 'count_season']

rdr = RandomForestRegressor(n_estimators=500,random_state=10)
rdr.fit(bike_train[registered_features],bike_train['registered_log'])
pred_registered = rdr.predict(bike_test[registered_features])
pred_registered = np.expm1(pred_registered)
pred_registered[pred_registered < 0 ] = 0

pred1 = pred_casual + pred_registered
submit_data=pd.DataFrame({'datetime':datetime,'count':pred1})
submit_data[submit_data['count']==0].count()
submit_data.to_csv("make_new_feature.csv", index=False)
def plot_cv(params,bestreg,variable):
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(params[variable],bestreg.cv_results_['mean_test_score'],'o-')
    plt.xlabel(variable)
    plt.ylabel('score mean')
    plt.subplot(122)
    plt.plot(params[variable],bestreg.cv_results_['std_test_score'],'o-')
    plt.xlabel(variable)
    plt.ylabel('score std')
    plt.tight_layout()
    plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
# create new_feature year_month for skfold
year_month = bike_train['year'] * 100 + bike_train['month']
skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
rdr = RandomForestRegressor(random_state=10)
params={'n_estimators':[50,100,150,200,250,300,350,400,450,500,550,600]}
bestreg = GridSearchCV(estimator=rdr,param_grid=params,cv=skfold.split(bike_train,year_month),scoring='neg_mean_squared_error')
bestreg.fit(bike_train[casual_features],bike_train['casual_log'])
print(bestreg.best_params_)
plot_cv(params,bestreg,'n_estimators')

rdr = RandomForestRegressor(random_state=10)
params={'n_estimators':[50,100,150,200,250,300,350,400,450,500,550,600]}
bestreg = GridSearchCV(estimator=rdr,param_grid=params,cv=skfold.split(bike_train,year_month),scoring='neg_mean_squared_error')
bestreg.fit(bike_train[registered_features],bike_train['registered_log'])
print(bestreg.best_params_)
plot_cv(params,bestreg,'n_estimators')
rdr = RandomForestRegressor(random_state=10)
params={'min_samples_leaf':np.arange(1,10,1)}
bestreg = GridSearchCV(estimator=rdr,param_grid=params,cv=skfold.split(bike_train,year_month),scoring='neg_mean_squared_error')
bestreg.fit(bike_train[casual_features],bike_train['casual_log'])
print(bestreg.best_params_)
plot_cv(params,bestreg,'min_samples_leaf')

rdr = RandomForestRegressor(random_state=10)
params={'min_samples_leaf':np.arange(1,10,1)}
bestreg = GridSearchCV(estimator=rdr,param_grid=params,cv=skfold.split(bike_train,year_month),scoring='neg_mean_squared_error')
bestreg.fit(bike_train[registered_features],bike_train['registered_log'])
print(bestreg.best_params_)
plot_cv(params,bestreg,'min_samples_leaf')
# rdr = RandomForestRegressor(random_state=10)
# params=[[{'n_estimators':[50,100,150,200,250,300,350,400,450,500,550,600]}],[{'min_samples_leaf':np.arange(1,10,1)}]]
# bestreg = GridSearchCV(estimator=rdr,param_grid=params,cv=skfold.split(bike_train,year_month),scoring='neg_mean_squared_error')
# bestreg.fit(bike_train[casual_features],bike_train['casual_log'])
# print(bestreg.best_params_)
# # plot_cv(params,bestreg,'n_estimators')

# rdr = RandomForestRegressor(random_state=10)
# params=[[{'n_estimators':[50,100,150,200,250,300,350,400,450,500,550,600]}],[{'min_samples_leaf':np.arange(1,10,1)}]]
# bestreg = GridSearchCV(estimator=rdr,param_grid=params,cv=skfold.split(bike_train,year_month),scoring='neg_mean_squared_error')
# bestreg.fit(bike_train[registered_features],bike_train['registered_log'])
# print(bestreg.best_params_)
# plot_cv(params,bestreg,'n_estimators')
# RMSLE Score 0.38262
casual_features=['season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed',
            'year', 'hour', 'dayofweek', 'hour_workingday_casual', 'count_season']

rdr = RandomForestRegressor(n_estimators=450,min_samples_leaf=5,random_state=10)
rdr.fit(bike_train[casual_features],bike_train['casual_log'])
pred_casual = rdr.predict(bike_test[casual_features])
pred_casual = np.expm1(pred_casual)
pred_casual[pred_casual < 0 ] = 0

registered_features=['season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed',
            'year', 'hour', 'dayofweek', 'hour_workingday_registered', 'count_season']

rdr = RandomForestRegressor(n_estimators=600,min_samples_leaf=3,random_state=10)
rdr.fit(bike_train[registered_features],bike_train['registered_log'])
pred_registered = rdr.predict(bike_test[registered_features])
pred_registered = np.expm1(pred_registered)
pred_registered[pred_registered < 0 ] = 0

pred1 = pred_casual + pred_registered
submit_data=pd.DataFrame({'datetime':datetime,'count':pred1})
submit_data[submit_data['count']==0].count()

submit_data.to_csv("make_new_feature_rdr.csv", index=False)
from sklearn.ensemble import GradientBoostingRegressor
gbrt_reg = GradientBoostingRegressor(random_state=10)
params={'n_estimators':[50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]}
bestreg = GridSearchCV(estimator=gbrt_reg,param_grid=params,cv=skfold.split(bike_train,year_month),scoring='neg_mean_squared_error')
bestreg.fit(bike_train[casual_features],bike_train['casual_log'])
print(bestreg.best_params_)
plot_cv(params,bestreg,'n_estimators')

gbrt_reg = GradientBoostingRegressor(random_state=10)
params={'n_estimators':[50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]}
bestreg = GridSearchCV(estimator=gbrt_reg,param_grid=params,cv=skfold.split(bike_train,year_month),scoring='neg_mean_squared_error')
bestreg.fit(bike_train[registered_features],bike_train['registered_log'])
print(bestreg.best_params_)
plot_cv(params,bestreg,'n_estimators')
gbrt_reg = GradientBoostingRegressor(random_state=10)
params={'min_samples_leaf':np.arange(1,10,1)}
bestreg = GridSearchCV(estimator=gbrt_reg,param_grid=params,cv=skfold.split(bike_train,year_month),scoring='neg_mean_squared_error')
bestreg.fit(bike_train[casual_features],bike_train['casual_log'])
print(bestreg.best_params_)
plot_cv(params,bestreg,'min_samples_leaf')

gbrt_reg = GradientBoostingRegressor(random_state=10)
params={'min_samples_leaf':np.arange(1,10,1)}
bestreg = GridSearchCV(estimator=gbrt_reg,param_grid=params,cv=skfold.split(bike_train,year_month),scoring='neg_mean_squared_error')
bestreg.fit(bike_train[registered_features],bike_train['registered_log'])
print(bestreg.best_params_)
plot_cv(params,bestreg,'min_samples_leaf')
# RMSLE Score 0.36942 rank 49/3251
casual_features=['season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed',
            'year', 'hour', 'dayofweek', 'hour_workingday_casual', 'count_season']

gbrt_reg = GradientBoostingRegressor(n_estimators=1000,min_samples_leaf=6,random_state=10)
gbrt_reg.fit(bike_train[casual_features],bike_train['casual_log'])
pred_casual = gbrt_reg.predict(bike_test[casual_features])
pred_casual = np.expm1(pred_casual)
pred_casual[pred_casual < 0 ] = 0

registered_features=['season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed',
            'year', 'hour', 'dayofweek', 'hour_workingday_registered', 'count_season']

gbrt_reg = GradientBoostingRegressor(n_estimators=1000,min_samples_leaf=5,random_state=10)
gbrt_reg.fit(bike_train[registered_features],bike_train['registered_log'])
pred_registered = gbrt_reg.predict(bike_test[registered_features])
pred_registered = np.expm1(pred_registered)
pred_registered[pred_registered < 0 ] = 0

pred2 = pred_casual + pred_registered
submit_data=pd.DataFrame({'datetime':datetime,'count':pred2})

submit_data.to_csv("make_new_feature_gbrt_reg.csv", index=False)
submit_data.shape
# RMSLE Score 0.37139 rank 88/3251

pred = 0.7 * pred1 + 0.3 * pred2
submit_data=pd.DataFrame({'datetime':datetime,'count':pred})

submit_data.to_csv("make_new_feature_rdr_70%_gbrt_reg_30%.csv", index=False)
# RMSLE Score 0.36714 rank 32/3251

pred = 0.5 * pred1 + 0.5 * pred2
submit_data = pd.DataFrame({'datetime':datetime,'count':pred})

submit_data.to_csv("make_new_feature_rdr_50%_gbrt_reg_50%.csv", index=False)