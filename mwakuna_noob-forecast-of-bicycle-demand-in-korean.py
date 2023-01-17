import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, make_scorer, r2_score

from sklearn.pipeline import Pipeline

from scipy.stats import norm, skew, kurtosis

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from xgboost import XGBRegressor



import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv", parse_dates=["datetime"])

sub = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")
print(train.shape)

print(train.columns)
print(test.shape)

print(test.columns)
del train['casual']

del train['registered']
print('===== train =====')

print(train.isnull().sum())

print('===== test =====')

print(test.isnull().sum())
train.info()
for val in ['season', 'holiday', 'workingday','weather']:

    train[val] = train[val].astype('category')

    test[val] = test[val].astype('category')
train['year'] = train['datetime'].dt.year

train['month'] = train['datetime'].dt.month

train['day'] = train['datetime'].dt.day

train['hour'] = train['datetime'].dt.hour

train['dayofweek'] = train['datetime'].dt.dayofweek
test['year'] = test['datetime'].dt.year

test['month'] = test['datetime'].dt.month

test['day'] = test['datetime'].dt.day

test['hour'] = test['datetime'].dt.hour

test['dayofweek'] = test['datetime'].dt.dayofweek
target = train['count']

sns.distplot(target, fit = norm)

plt.annotate('skewness: {0}'. format(np.round(skew(target),3)), xy=(20,0.006))

plt.annotate('kurtosisness: {0}'.format(np.round(kurtosis(target),3)),xy = (20,0.0065))
def scaler(x):

    return ( x - np.mean(x)) / np.std(x)



sns.distplot(scaler(target))

print('mean :',np.mean(scaler(target)))

print('std  :',np.std(scaler(target)))

plt.annotate('skewness: {0}'. format(np.round(skew(scaler(target)),3)),

             xy =(2,1.05))

plt.annotate('kurtosisness: {0}'. format(np.round(kurtosis(scaler(target)),3)),

             xy =(2,1))
sns.distplot(np.log1p(target), fit = norm)



plt.annotate('skewness: {0}'.format(np.round(skew(np.log1p(target)),3)), xy = (1,0.35))

plt.annotate('kurtosisness: {0}'.format(np.round(kurtosis(np.log1p(target)),3)), xy = (1,0.32))
train['count']  = np.log1p(target)
sns.pairplot(train[['temp','atemp','humidity', 'windspeed']])
train[['temp','atemp','count']].corr(method='pearson')
del train['atemp']

del test['atemp']
for val in ['windspeed','humidity','temp']:

    print ('{}` skewness : {:.3f}'.format(val, skew(train[val])))
sns.boxplot(x = train['year'],

           y = train['count'])
sns.boxplot(x = train['season'],

           y = train['count'])
sns.boxplot(x = train['hour'],

           y = train['count'])
for i in [0,1,2,3,4] :

    

    train['hour'].replace(i, i+24, inplace = True)

    test['hour'].replace(i, i+24, inplace = True)

    

train.hour.value_counts()
plt.scatter(x= train['hour'],

           y = train['count'],

           alpha = 0.3,

           color = 'blue')



plt.xlabel('hour')

plt.ylabel('count')



plt.title('hour ~ count')



sns.regplot(x = train['hour'],

            y = train['count'],

           order = 2 , label = 'degree.2')

sns.regplot(x = train['hour'],

            y = train['count'],

           order = 1 , label = 'degree.1')



plt.legend(loc = 'upper right')

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import RandomForestClassifier



lr_m = LinearRegression()

poly_m = LinearRegression()

rfc = RandomForestClassifier()



train_hour = train.loc[:, 'hour'].values

test_hour = test.loc[:, 'hour'].values





train_hour = train_hour.reshape(-1, 1)

test_hour = test_hour.reshape(-1, 1)



poly = PolynomialFeatures(degree=2)



train_hour_sqr = poly.fit_transform(train_hour)

test_hour_sqr = poly.fit_transform(test_hour)



lr_m.fit(train_hour, target)

poly_m.fit(train_hour_sqr, target)



pred_lr = lr_m.predict(train_hour)

pred_poly = poly_m.predict(train_hour_sqr)



lr_score = mean_squared_error(target, pred_lr)

poly_score = mean_squared_error(target, pred_poly)



r2_lr = r2_score(target, pred_lr)

r2_poly = r2_score(target, pred_poly)



print('r2_score (degree = 1) : {0:.3f} \n  MSE : {1:.3f}'

      .format(r2_lr, lr_score))

print('======================================================')

print('r2_score(degree = 2) : {0:.3f} \n MSE : {1:.3f}'

      .format(r2_poly, poly_score))

print('======================================================')

print('polynomial regression estimators : ({1:.3f}) * hour^2 + ({0:.3f}) * hour + ({2:.3f})'

      .format(poly_m.coef_[1],poly_m.coef_[2], poly_m.intercept_))
train['hour_sqr'] = np.square(train['hour'])

test['hour_sqr'] = np.square(test['hour'])
train.holiday.value_counts()
train.loc[train.dayofweek >= 5,'holiday'] = 1

test.loc[test.dayofweek >= 5, 'holiday'] = 1
train.holiday.value_counts()
sns.boxplot(x = train['holiday'],

            y = train['count'])
plt.figure(figsize = (10,10))

sns.boxplot(x = train['hour'],

            y = train['count'],

            hue = train['holiday'])
sns.boxplot(x = train['month'],

            y = train['count'])
plt.scatter(x = train['month'],

            y = train['count'])
plt.scatter(x = train['windspeed'],

            y = train['count'])

plt.xlabel('windspeed')

plt.ylabel('count')
sns.boxplot(x = train['weather'],

            y = train['count'])
print('train \n',train.weather.value_counts())

print('test \n',test.weather.value_counts())
train['weather'].replace(3,4,inplace = True)

test['weather'].replace(3,4,inplace = True)
plt.scatter(x = train['humidity'],

            y = train['count'],

           alpha = 0.5,

           color = 'blue')

plt.xlabel('humidity')

plt.ylabel('count')



plt.annotate('correlation between humidity and count : {0:.3f}'.format(train[['humidity','count']].corr().iloc[1,0]), xy = (0,6))

sns.regplot(x = train['humidity'],

            y = train['count'])
sns.boxplot(x = train['dayofweek'],

            y = train['count'])
plt.scatter(x = train['hour'],

            y = train['count'],

            c = train['dayofweek'])
plt.scatter(x = train['temp'],

            y = train['count'])



plt.annotate('correlation between humidity and count : {0:.3f}'.format(train[['temp','count']].corr().iloc[1,0]), xy = (0,6))

sns.regplot(x = train['temp'],

            y = train['count'])
sns.heatmap(train[['temp','windspeed','humidity','hour','count']].corr(method = 'pearson'),annot = True, fmt = '.2f')
train.info()
del train['datetime']

del test['datetime']
def rmsle(y, pred):

    log_y=np.log1p(y+1)

    log_pred=np.log1p(pred + 1)

    squared_error=(log_y-log_pred)**2

    rmsle=np.sqrt(np.mean(squared_error))

    return rmsle



rmsle_score = make_scorer(rmsle)
target = train['count']

train.drop('count', axis = 1, inplace = True) # target, feature 분리
rf_reg = RandomForestRegressor(n_estimators=1000, n_jobs = -1,

                               random_state = 777)

# 1000개의 의사결정 나무 생성, cpu 집중, 난수 고정



x_train, x_test, y_train, y_test = train_test_split(train,

                                                    target,

                                                    test_size = 0.3,

                                                    random_state = 777)



rf_reg.fit(x_train, y_train)



pred = rf_reg.predict(x_test)



pred = np.expm1(pred)

y_test = np.expm1(y_test)



print('RandomForest score : ', rmsle(pred, y_test))
feat_imp = {'col' : train.columns,

            'importances' : rf_reg.feature_importances_}



feat_imp = pd.DataFrame(feat_imp).sort_values(by = 'importances', ascending = False)



sns.barplot(x = feat_imp['col'] ,

            y = feat_imp['importances'])

plt.xticks(rotation =  60)
print(feat_imp)
pred = rf_reg.predict(test)

pred = np.expm1(pred)

sub['count'] = pred

sub.to_csv('submission_20200202.csv',index = False)
for val in ['weather','day','windspeed','holiday']:

    del train[val]

    del test[val]
train_m = pd.get_dummies(train, columns=['month','year','dayofweek','workingday','season'])

test_m = pd.get_dummies(test,columns=['month','year','dayofweek','workingday','season'])

print(train_m.shape)

print(test_m.shape)
rf_reg=RandomForestRegressor()

xgb_reg=XGBRegressor()

lr_reg=LinearRegression()

lasso=Lasso()

ridge=Ridge()

elastic=ElasticNet()



model_list=[rf_reg, xgb_reg, lr_reg, ridge, lasso , elastic]

for model in model_list :

    score = cross_val_score(model, train_m, target , scoring = rmsle_score, cv = 3)

    print('{0} ` score : {1}'.format(model.__class__.__name__, np.mean(score)))
params = {'max_depth': [3,5,7,11],

         'min_samples_split': [2,4,6,8],

         "min_weight_fraction_leaf": [0.01,0.1,0.2,0.3],

         "max_features":[4,5,6]}

grid_rf = GridSearchCV(rf_reg,param_grid = params, n_jobs=-1)

grid_rf.fit(train_m, target)
print(grid_rf.best_params_)
pred = grid_rf.predict(test_m)



sub['count'] = np.expm1(pred)



sub.to_csv('bike_submission_noob.csv', index = False)