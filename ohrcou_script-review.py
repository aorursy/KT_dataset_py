import numpy as np

import pandas as pd

import xgboost as xgb

import lightgbm as lgb

from sklearn import neighbors

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

import random

import datetime as dt

import gc



#Viz

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

from numpy import arange

%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



#Explore training data

train_start= pd.read_csv('../input/train_2016_v2.csv', parse_dates=["transactiondate"])

train_start.head()



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Explore training data

train_start= pd.read_csv('../input/train_2016_v2.csv', parse_dates=["transactiondate"])

train_start.head()
#Explore LogError to check for outliers because improving residual error is a component of the competition

plt.figure(figsize=(8,6))

plt.scatter(range(train_start.shape[0]), np.sort(train_start.logerror.values))

plt.xlabel('index', fontsize=14)

plt.ylabel('logerror', fontsize=14)

plt.show()
#Remove outliers

#upper_per = np.percentile(train_start.logerror.values, 99)

#lower_per = np.percentile(train_start.logerror.values, 1)

#train_start['logerror'].ix[train_start['logerror']>upper_per] = upper_per

#train_start['logerror'].ix[train_start['logerror']<lower_per] = lower_per



#Histogram of resulting data

#plt.figure(figsize=(12,8))

#sns.distplot(train_start.logerror.values, bins=50, kde=False)

#plt.xlabel('logerror', fontsize=14)

#plt.show()
#Explore the data sales dates i.e. number of transactions in a month

train_start['transaction_month']=train_start['transactiondate'].dt.month

counts=train_start['transaction_month'].value_counts()

plt.figure(figsize=(14,8))

sns.barplot(counts.index, counts.values, alpha=0.8)

plt.xticks(rotation='vertical')

plt.xlabel('Transactions (month)', fontsize=14)

plt.ylabel('Occurences',fontsize=14)

plt.show()
#Datasets

prop_start = pd.read_csv('../input/properties_2016.csv')

test2 = pd.read_csv('../input/sample_submission.csv')

#Rename test data field ParcelID to match training data

test = test2.rename(columns={'ParcelId':'parcelid'})

#print (test.dtypes)

#print (train.dtypes)

#print (props.dtypes)



#Explore Propery data fields

prop_start.head()
##Change the data types to improve memory usage

for column in prop_start.columns:

    if prop_start[column].dtype==int:

        prop_start[column]=prop_start[column].astype(np.int32)

    if prop_start[column].dtype==float:

        prop_start[column]=prop_start[column].astype(np.float32)

        

for column in test.columns:

       if test[column].dtype==int:

           test[column]=test[column].astype(np.int32)

       if test[column].dtype==float:

           test[column]=test[column].astype(np.float32)
#Visualize the missing data NaN

missing_dv=prop_start.isnull().sum(axis=0).reset_index()

missing_dv.columns=['column_name', 'missing']

missing_dv = missing_dv.ix[missing_dv['missing']>0]

missing_dv = missing_dv.sort_values(by='missing')

wth=0.9

index=np.arange(missing_dv.shape[0])

fig, ax = plt.subplots(figsize=(14,20))

rects = ax. barh(index, missing_dv.missing.values, color='blue')

ax.set_yticks(index)

ax.set_yticklabels(missing_dv.column_name.values, rotation='horizontal')

ax.set_xlabel("Missing Values (count)")

plt.show()
#Explore spatial data files

plt.figure(figsize=(14,14))

sns.jointplot(x=prop_start.latitude.values, y=prop_start.longitude.values, size=12)

plt.ylabel('Longitude',fontsize=8)

plt.xlabel('Latitude', fontsize=8)

plt.show()
###Calculate some properties for estimating home value



#Living area of properties

prop_start['living_area'] = prop_start['calculatedfinishedsquarefeet']/prop_start['lotsizesquarefeet']

prop_start['EstRoomSize']= prop_start['calculatedfinishedsquarefeet']/prop_start['roomcnt']



#Ratio of tax per worth of property

prop_start['tax_ratio']=prop_start['taxvaluedollarcnt']/prop_start['taxamount']



#Proportion of structure value to land

prop_start['tax_proportion']=prop_start['structuretaxvaluedollarcnt']/prop_start['landtaxvaluedollarcnt']



#Time of unpaid taxes

prop_start['DeliqCount']= 2018- prop_start['taxdelinquencyyear']



#Simplify Land Uses from 25 to 4 categories: Mixed, Home, Other, Not Built

prop_start['LandUse'] = prop_start.propertylandusetypeid.replace({31 : "Mixed", 46 : "Other", 

                                                                  47 : "Mixed", 246 : "Mixed", 

                                                                  247 : "Mixed", 248 : "Mixed", 

                                                                  260 : "Home", 261 : "Home", 262 : "Home", 

                                                                  263 : "Home", 264 : "Home", 265 : "Home", 

                                                                  266 : "Home", 267 : "Home", 268 : "Home", 

                                                                  269 : "Not Built", 270 : "Home", 271 : "Home", 

                                                                  273 : "Home", 274 : "Other", 275 : "Home", 

                                                                  276 : "Home", 279 : "Home", 290 : "Not Built", 

                                                                  291 : "Not Built" })

print('Done')
#Add propertiy data to training and testing datasets

m_train= train_start.merge(prop_start, how='left', on='parcelid')

m_test= test.merge(prop_start, how='left', on='parcelid')



#Look at heatmap to see if variables are correlated

var= ['airconditioningtypeid','architecturalstyletypeid','buildingqualitytypeid',

            'buildingclasstypeid','decktypeid','fips','hashottuborspa','heatingorsystemtypeid','living_area',

            'pooltypeid10','pooltypeid2','pooltypeid7','propertycountylandusecode',

            'propertylandusetypeid','propertyzoningdesc','rawcensustractandblock','regionidcity',

            'regionidcounty','regionidneighborhood','regionidzip','storytypeid','tax_ratio', 'tax_proportion',

            'typeconstructiontypeid','yearbuilt','taxdelinquencyflag']

call=[i for i in m_train.columns if i not in var]

plt.figure(figsize=(14,14))

cmap=sns.diverging_palette(220,20, sep=20, as_cmap=True)

sns.heatmap(data=m_train[call].corr(),cmap=cmap)

plt.show()

plt.gcf().clear()
###More data clean-up, identify and fill-in missing values with label encoder

#Label Encoding: Encode labels with value between 0 and n_classes-1.

from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()

for l in m_train.columns:

    m_train[l]=m_train[l].fillna(0)

    if m_train[l].dtype=='object':

        LE.fit(list(m_train[l].values)) #normalize labels

        m_train[l]=LE.transform(list(m_train[l].values))  #transform non-numerical labels to numerical

        

for l in m_test.columns:

    m_test[l]=m_test[l].fillna(0)

    if m_test[l].dtype=='object':

        LE.fit(list(m_test[l].values)) #normalize labels

        m_test[l]=LE.transform(list(m_test[l].values)) #transform non-numerical labels to numerical

        

#Drop properties we don't need anymore for training: parcelid, log error, sell date, propert zoning description and land use code

sec_train=m_train.drop(['transaction_month','parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)

#Drop values we don't need for validation

sec_test= m_test.drop(['parcelid','propertyzoningdesc','propertycountylandusecode','201610','201611','201612', '201710', '201711', '201712'], axis=1)

print('Done')
#Use XGBoost to look more into the variable importance

X=sec_train.values

Y=m_train['logerror'].values

xgb_params = {'eta': 0.05,'max_depth': 8,'subsample': 0.7,'colsample_bytree': 0.7, 'objective': 'reg:linear',

    'silent': 1,'seed' : 0}

dtrain = xgb.DMatrix(sec_train, Y, feature_names=sec_train.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=150)

# plot the important features #

fig, ax = plt.subplots(figsize=(18,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
train_columns = sec_train.columns

d_train = lgb.Dataset(sec_train, label=Y)

params = {'max_bin': 10,'learning_rate': 0.0021,'boosting_type': 'gbdt','objective': 'regression',

          'metric': 'l1','sub_feature': 0.345,'bagging_fraction': 0.85,'bagging_freq': 40,

          'num_leaves':512, 'min_data': 500,'min_hessian': 0.05,'verbose': 0,'feature_fraction_seed':2,

          'bagging_seed': 3}

#Light Gradient Boosting Model

print('Fitting LightGBM model ...')

clf = lgb.train(params, d_train, 200)

print ('completed')

Xtest = sec_test[train_columns]

p_test = clf.predict(Xtest)

#print('1')

Ymean = np.mean(Y)

#Extreme Gradient Boosting Model 1

print ('prepping paramaters for Xboost')

paramsXGB = {'eta': 0.037,'max_depth': 5,'subsample': 0.80,'objective': 'reg:linear','eval_metric': 'mae',

    'lambda': 0.8,'alpha': 0.4, 'base_score': Ymean,'silent': 1}

dtrain = xgb.DMatrix(sec_train, Y)

dtest = xgb.DMatrix(Xtest)

num_boost = 250

print( 'Training XGBoost')

model = xgb.train(dict(paramsXGB, silent=1), dtrain, num_boost_round=num_boost)

print( 'Predicting with XGBoost')

xgb_pred1 = model.predict(dtest)

print( 'First XGBoost predictions:')

print( pd.DataFrame(xgb_pred1).head())
#Extreme Gradient Boosting Model 2

paramsXGB2 = {'eta': 0.033,'max_depth': 6,'subsample': 0.80,'objective': 'reg:linear',

    'eval_metric': 'mae','base_score': Ymean,'silent': 1}

num_boost2 = 150

print( 'Training XGBoost for second model ...')

model = xgb.train(dict(paramsXGB2, silent=1), dtrain, num_boost_round=num_boost2)

#print( 'Predicting with XGBoost second time')

xgb_pred2 = model.predict(dtest)

print( 'Second XGBoost predictions:' )

print( pd.DataFrame(xgb_pred2).head() )
XGB1_WEIGHT = 0.8083 # Weight of first in combination of two XGB models

xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2

print( 'Combined XGBoost predictions:' )

print( pd.DataFrame(xgb_pred).head() )
#Combine LGB and Xboost

XGB_WEIGHT = 0.700 #based on https://www.kaggle.com/hsperr/finding-ensamble-weights and various Kernel results

BASELINE_WEIGHT = 0.0056 #based on

OLS_WEIGHT = 0.0620 # based on https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach

XGB1_WEIGHT = 0.8083 # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115 # Baseline based on mean of training data

gc.collect()

np.random.seed(17)

random.seed(17)

train_F = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

properties2 = pd.read_csv("../input/properties_2016.csv")

submission = pd.read_csv("../input/sample_submission.csv")

#print(len(train_F),len(properties2),len(submission))

##OLS based on https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach

def get_features(df):

    df["transactiondate"] = pd.to_datetime(df["transactiondate"])

    df["transactiondate_year"] = df["transactiondate"].dt.year

    df["transactiondate_month"] = df["transactiondate"].dt.month

    df['transactiondate'] = df['transactiondate'].dt.quarter

    df = df.fillna(-1.0)

    return df

#Submission Calculation: logerror=log(Zestimate)−log(SalePrice)

def MAE(y, ypred):

    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

train_F = pd.merge(train_F, properties2, how='left', on='parcelid')

Y = train_F['logerror'].values

test = pd.merge(submission, properties2, how='left', left_on='ParcelId', right_on='parcelid')

properties = [] #memory

exc = [train_F.columns[c] for c in range(len(train_F.columns)) if train_F.dtypes[c] == 'O'] + ['logerror','parcelid']

col = [c for c in train_F.columns if c not in exc]

train_F = get_features(train_F[col])

test['transactiondate'] = '2016-01-01' 

test = get_features(test[col])

reg = LinearRegression(n_jobs=-1)

reg.fit(train_F, Y); #print('fit...')

#print(MAE(Y, reg.predict(train_F)))

train_F = [];  Y = [] 

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']

test_columns = ['201610','201611','201612','201710','201711','201712']

print( 'Combining XGBoost, LightGBM, and baseline predicitons' )

lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)

xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)

baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)

pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test

print( 'Combined XGB/LGB/baseline predictions:' )

print( pd.DataFrame(pred0).head() )

print( 'Predicting with OLS and combining with XGB/LGB/baseline predicitons:' )

for i in range(len(test_dates)):

    test['transactiondate'] = test_dates[i]

    pred = OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0

    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]

    print('predict...', i)

print( 'Final Prediction Log Errors (XGB/LGB/baseline):' )

from datetime import datetime

print( 'Writing results to disk')

submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( 'Finished')

LogError=submission['201610'].values

Avg=np.mean(LogError)

print ('Avgerage Log Error: ')

print (Avg)

submission