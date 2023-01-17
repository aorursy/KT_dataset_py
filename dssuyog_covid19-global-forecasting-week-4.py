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
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
samplesub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
# from pandas_profiling import ProfileReport
# ProfileReport(train)
train.dtypes
train.head()
test.head()
samplesub.head()
train.isna().sum()
# [code: 103]
train.Province_State.fillna(train.Country_Region, inplace=True)
train.Province_State.isna().sum()
train[['Province_State', 'Country_Region', 'Id']] = train[['Province_State', 'Country_Region', 'Id']].apply(lambda x: x.astype('category'))
train[['ConfirmedCases', 'Fatalities']] = train[['ConfirmedCases', 'Fatalities']].astype('int')

train['Date']= pd.to_datetime(train['Date'])
# Or, train.Date = train.Date.apply(pd.to_datetime)
train.dtypes
train.head()
train.isna().sum() / train.shape[0] * 100
# [code: 103] one line code above for same code below
# train['Province_State'] = train['Province_State'].astype('str')
# train['Country_Region'] = train['Country_Region'].astype('str')

# # train.loc[train['Province_State'] == 'nan', ['Province_State']].shape
# # train.loc[train['Province_State'] == 'nan', ['Country_Region']].shape
# # train['Province_State'].isna().sum()

# train.loc[train['Province_State'] == 'nan', ['Province_State']] = train.loc[train['Province_State'] == 'nan', ['Country_Region']]
# train['Province_State'].fillna(train['Country_Region'], inplace=True) # category not present

# # train['Province_State'] = train['Province_State'].astype('category')
# train['Province_State'] = pd.Categorical(train['Province_State'])
# train['Country_Region'] = pd.Categorical(train['Country_Region'])
train['days_since'] = train['Date'].apply(lambda x: (x - pd.to_datetime('2020-01-21')).days )
train[train['days_since']==1].head()
# train['month_date'] = train.Date.dt.strftime("%m%d").astype(int)
# Adding feature continent and sub_region

coucon = pd.read_csv('/kaggle/input/country-to-continent/countryContinent.csv', encoding='latin-1')

# ss = train.merge(coucon[['country', 'continent', 'sub_region']],how='left', left_on='Country_Region', right_on='country').fillna(np.nan)
# train.drop(['country'],axis=1,inplace=True)
# ss[ss['continent'].isna()]['Country_Region'].unique()

def get_continent(x):
    if coucon['country'].str.contains(x).any():
        return coucon.loc[coucon['country'].str.contains(x), 'continent'].iloc[0] 
    else:
        np.nan

train['continent'] = train['Country_Region'].apply(get_continent)

train.loc[train['Country_Region']== 'Burma', ['continent']] = 'Asia'
train.loc[train['Country_Region'].isin(['Congo (Brazzaville)', 'Congo (Kinshasa)']), ['continent']] = 'Africa'
train.loc[train['Country_Region']== "Cote d'Ivoire", ['continent']] = 'Africa'
train.loc[train['Country_Region']== "Czechia", ['continent']] = 'Europe'
train.loc[train['Country_Region']== "Diamond Princess", ['continent']] = 'Asia'
train.loc[train['Country_Region']== "Eswatini", ['continent']] = 'Africa'
train.loc[train['Country_Region']== "India", ['continent']] = 'Asia'
train.loc[train['Country_Region']== "Korea, South", ['continent']] = 'Asia'
train.loc[train['Country_Region']== "Kosovo", ['continent']] = 'Europe'
train.loc[train['Country_Region']== "Laos", ['continent']] = 'Asia'
train.loc[train['Country_Region']== "MS Zaandam", ['continent']] = 'Americas'
train.loc[train['Country_Region']== "North Macedonia", ['continent']] = 'Europe'
train.loc[train['Country_Region']== "US", ['continent']] = 'Americas'
train.loc[train['Country_Region']== "Vietnam", ['continent']] = 'Asia'
train.loc[train['Country_Region']== "West Bank and Gaza", ['continent']] = 'Asia'

train.continent = train.continent.astype('category')

train['continent'].isna().sum()
train.dtypes
trainc = train.copy()
# y = trainc.loc[:, ['ConfirmedCases']]
# y = trainc.loc[:, ['Fatalities']]
y = trainc.loc[:, ['ConfirmedCases', 'Fatalities']]

# X = trainc.loc[:, ['Province_State', 'Country_Region', 'Date', 'days_since']]
X = trainc.loc[:, ['Province_State', 'Country_Region', 'days_since',]]

# split dataset in train and test
from sklearn.model_selection import train_test_split
# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Merge less frequent levels. OHE on limited levels won't generate large no. of features.

# Define which columns should be encoded vs scaled
numeric_ftrs  = X_train.select_dtypes(include=['float64','int64'])
categorical_ftrs = X_train.select_dtypes(include=['category'])

# Instantiate encoder/scaler
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)

# auto
scaled_data = pd.DataFrame(scaler.fit_transform(numeric_ftrs), columns = ['days_since'])

encoder = OneHotEncoder(handle_unknown="ignore")
# encoder.fit(X_train)
encoder.fit(categorical_ftrs)
# encoded_ftrs = pd.DataFrame(encoder.fit_transform(categorical_ftrs).toarray())
encoded_ftrs = pd.DataFrame(encoder.fit_transform(categorical_ftrs).toarray(), columns = encoder.get_feature_names(categorical_ftrs.columns.to_list()))

X_train = pd.concat([scaled_data, encoded_ftrs], axis = 1)
print("ran")
te_numeric_ftrs  = X_test.select_dtypes(include=['float64','int64'])
te_categorical_ftrs = X_test.select_dtypes(include=['category'])

te_scaled_data = pd.DataFrame(scaler.fit_transform(te_numeric_ftrs), columns = numeric_ftrs.columns.to_list())

# te_encoded_ftrs = pd.DataFrame(encoder.fit_transform(te_categorical_ftrs).toarray())
te_encoded_ftrs = pd.DataFrame(encoder.fit_transform(te_categorical_ftrs).toarray(), columns = encoder.get_feature_names(te_categorical_ftrs.columns.to_list()))

X_test = pd.concat([te_scaled_data, te_encoded_ftrs], axis = 1)
print("ran")
rf_y_train = y_train.copy()
rf_y_test = y_test.copy()

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_jobs=-1, n_estimators=100, oob_score=True, max_features='sqrt')
# rf = RandomForestRegressor(n_jobs=-1,n_estimators=50,oob_score=True,max_features=0.8,min_samples_leaf=5)
# rf = RandomForestRegressor(n_jobs=-1,n_estimators=50,min_samples_leaf=7,max_features=0.5,oob_score=True,max_depth=40)

rf.fit(X_train, rf_y_train)

y_train_cc = y_train['ConfirmedCases']
y_train_fat = y_train['Fatalities']
# # XGBoost

# from sklearn.model_selection import ShuffleSplit, cross_val_score
# skfold = ShuffleSplit(random_state=7)

# import xgboost as xgb

# reg_xgb_cc = xgb.XGBRegressor(n_estimators = 400)
# reg_xgb_fat = xgb.XGBRegressor(n_estimators = 200)

# xgb_acc = cross_val_score(reg_xgb_cc, X_train, y_train_cc, cv = skfold)
# xgb_acc_fat = cross_val_score(reg_xgb_fat, X_train, y_train_fat, cv = skfold)

# # print (xgb_acc.mean())
# print (xgb_acc.mean(), xgb_acc_fat.mean())
# reg_xgb_cc.fit(X_train, Y_train_cc)
# y_pred_cc = reg_xgb_cc.predict(X_test) 

# reg_xgb_fat.fit(X_train, Y_train_fat)
# y_pred_fat = reg_xgb_fat.predict(X_test) 

# # BaggingRegressor
# y_train_cc = y_train['ConfirmedCases']
# y_train_fat = y_train['Fatalities']

# from sklearn.ensemble import BaggingRegressor
# from sklearn.tree import DecisionTreeRegressor

# reg_bgr_cc = BaggingRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 100)
# reg_bgr_fat = BaggingRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 70)

# bgr_acc = cross_val_score(reg_bgr_cc, X_train, y_train_cc, cv = skfold)
# bgr_acc_fat = cross_val_score(reg_bgr_fat, X_train, y_train_fat, cv = skfold)
# print (bgr_acc.mean(), bgr_acc_fat.mean())
# reg_bgr_cc.fit(X_train, y_train_cc)
# y_pred_cc = clf_bgr_cc.predict(X_test)

# reg_bgr_fat.fit(X_train, y_train_cc)
# y_pred_cc = reg_bgr_fat.predict(X_test) 

print('train Score: ', rf.score(X_train, rf_y_train), ". ", end = '')
print('test Score: ', rf.score(X_test, rf_y_test))
# Chained Models for Each Output (RegressorChain)
# https://machinelearningmastery.com/multi-output-regression-models-with-python/
# Another approach to using single-output regression models for multioutput regression is to create a linear 
# sequence of models.

# The first model in the sequence uses the input and predicts one output; the second model uses the input and 
# the output from the first model to make a prediction; the third model uses the input and output from the 
# first two models to make a prediction, and so on.

from sklearn.multioutput import RegressorChain
wrapper = RegressorChain(rf)
wrapper.fit(X_train, y_train)

rf_y_test_pred = wrapper.predict(X_test)
# summarize prediction
print(rf_y_test_pred[0:5])
print(rf_y_test_pred.astype('int')[0:5])
rf_y_test_pred = rf_y_test_pred.astype('int')
# Use the R forest's predict method on the test data
rf_y_test_pred = rf.predict(X_test)
print(rf_y_test_pred[0:5])
print(rf_y_test_pred.astype('int')[0:5])
rf_y_test_pred = rf_y_test_pred.astype('int')
# RMSLE
from sklearn.metrics import mean_squared_log_error
# np.sqrt(mean_squared_log_error(y_test, predictions ))
rf_y_test
rf_y_test_pred2 = pd.DataFrame(rf_y_test_pred, columns = ['ConfirmedCases', 'Fatalities'])

np.sqrt(mean_squared_log_error(rf_y_test['ConfirmedCases'], rf_y_test_pred2['ConfirmedCases']))
np.sqrt(mean_squared_log_error(rf_y_test['Fatalities'], rf_y_test_pred2['Fatalities']))

# mean of individual rmsle
res = (np.sqrt(mean_squared_log_error(rf_y_test['ConfirmedCases'], rf_y_test_pred2['ConfirmedCases'])) +
np.sqrt(mean_squared_log_error(rf_y_test['Fatalities'], rf_y_test_pred2['Fatalities'])) )/2
print("mean of rmsle individual (func) = {}".format(res))


import math
#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y)) ) ** 0.5

res = (rmsle(rf_y_test['ConfirmedCases'].to_numpy(), rf_y_test_pred2['ConfirmedCases'].to_numpy()) +
rmsle(rf_y_test['Fatalities'].to_numpy(), rf_y_test_pred2['Fatalities'].to_numpy()) )/2

print("mean of rmsle individual(manual) = {}".format(res))


# RMSLE combined for target columns
# first sum of squared errors, then mean and root
# manual
# y,y_pred = rf_y_test['ConfirmedCases'].to_numpy(), rf_y_test_pred2['ConfirmedCases'].to_numpy()
# conf_sle2 = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

# y,y_pred = rf_y_test['Fatalities'].to_numpy(), rf_y_test_pred2['Fatalities'].to_numpy()
# fata_sle2 = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

# print( ( (sum(conf_sle2) + sum(fata_sle2)) * (1.0/(2 * len(y))) )** 0.5 )


def rmsle_2col(y1, y_pred1,y2, y_pred2):
    assert len(y1) == len(y_pred1)
    terms_to_sum1 = [(math.log(y_pred1[i] + 1) - math.log(y1[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred1)]
    assert len(y2) == len(y_pred2)
    terms_to_sum2 = [(math.log(y_pred2[i] + 1) - math.log(y2[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred2)]
    
    return ( (sum(terms_to_sum1) + sum(terms_to_sum2) ) * (1.0/(2 * len(y1))) ) ** 0.5

res = rmsle_2col(rf_y_test['ConfirmedCases'].to_numpy(), rf_y_test_pred2['ConfirmedCases'].to_numpy(),rf_y_test['Fatalities'].to_numpy(), rf_y_test_pred2['Fatalities'].to_numpy())
print("RMSLE combined = {}".format(res))
# MTR  both  RF-100 estimators
# mean of rmsle individual (func) = 0.368470396754244
# mean of rmsle individual(manual) = 0.3684703967542432
# RMSLE combined = 0.3749458290992757

# MTR chained regression RF-100 estimators
# mean of rmsle individual (func) = 0.347249992389988
# mean of rmsle individual(manual) = 0.347249992389988
# RMSLE combined = 0.36388949014719607



# [code 103]
test.Province_State.fillna(train.Country_Region, inplace=True)

test[['Province_State', 'Country_Region', 'ForecastId']] = test[['Province_State', 'Country_Region', 'ForecastId']].apply(lambda x: x.astype('category'))
test['Date']= pd.to_datetime(test['Date'])
# replaced [code 103]

# test['Province_State'] = test['Province_State'].astype('str')
# test['Country_Region'] = test['Country_Region'].astype('str')

# # test.loc[test['Province_State'] == 'nan', ['Province_State']].shape
# # test.loc[test['Province_State'] == 'nan', ['Country_Region']].shape
# # test['Province_State'].isna().sum()

# test.loc[test['Province_State'] == 'nan', ['Province_State']] = test.loc[test['Province_State'] == 'nan', ['Country_Region']]
# test['Province_State'].fillna(test['Country_Region'], inplace=True) # category not present

# # test['Province_State'] = test['Province_State'].astype('category')
# test['Province_State'] = pd.Categorical(test['Province_State'])
# test['Country_Region'] = pd.Categorical(test['Country_Region'])
test['days_since'] = test['Date'].apply(lambda x: (x - pd.to_datetime('2020-01-21')).days )
test[test['days_since']>1].head()
testc = test.copy()
# y = testc.loc[:, ['ConfirmedCases']]
X_score = testc.loc[:, ['Province_State', 'Country_Region', 'days_since']]
score_numeric_ftrs  = X_score.select_dtypes(include=['float64','int64'])
score_categorical_ftrs = X_score.select_dtypes(include=['category'])

score_scaled_data = pd.DataFrame(scaler.fit_transform(score_numeric_ftrs), columns = ['days_since'])

# score_encoded_ftrs = pd.DataFrame(encoder.fit_transform(score_categorical_ftrs).toarray())
score_encoded_ftrs = pd.DataFrame(encoder.fit_transform(score_categorical_ftrs).toarray(), columns = encoder.get_feature_names(score_categorical_ftrs.columns.to_list()))

X_score = pd.concat([score_scaled_data, score_encoded_ftrs], axis = 1)
print("ran")
# Use the R forest's predict method on the test data
rf_y_score = rf.predict(X_score)
print(rf_y_score[0:5])
print(rf_y_score.astype('int')[0:5])
rf_y_score = rf_y_score.astype('int')
# sub = pd.concat([test['ForecastId'], pd.DataFrame(rf_y_score, columns = ['ConfirmedCases']), ], axis = 1)
# sub = pd.concat([test['ForecastId'], pd.DataFrame(rf_y_score, columns = ['Fatalities']), ], axis = 1)
sub = pd.concat([test['ForecastId'], pd.DataFrame(rf_y_score, columns = ['ConfirmedCases', 'Fatalities']), ], axis = 1)
sub.head()


# sub.to_csv('/kaggle/working/sub_conf.csv', index = False)
# sub.to_csv('/kaggle/working/sub_fata.csv', index = False)
# sub.to_csv('/kaggle/working/sub_both.csv', index = False)

os.listdir('/kaggle/working/')

# conf = pd.read_csv('/kaggle/working/sub_conf.csv')
# fata = pd.read_csv('/kaggle/working/sub_fata.csv')
# both = pd.read_csv('/kaggle/working/sub_both.csv')

# both.to_csv('/kaggle/working/samplesubmission.csv', index = False)

# all = pd.concat([conf, both, fata], axis = 1)

# sub = pd.read_csv('/kaggle/working/sub_both.csv')
# sub.to_csv('samplesubmission.csv', index = False)

# print("End..")
# print(os.listdir('/kaggle/working/'))

