import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_log_error



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import sys

np.set_printoptions(threshold=sys.maxsize)

import warnings

warnings.filterwarnings('ignore')
train= pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

test= pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
def explore(df):

    checkNulls= df.isna().sum()

    checkNullsPercentage= (checkNulls/df.shape[0]) * 100

    return pd.DataFrame({'Missing Values': checkNulls, "Data Type":df.dtypes, 

                         "No of Levels":df.apply(lambda x: x.nunique(),axis=0), 

                         "Levels":df.apply(lambda x: str(x.unique()),axis=0)})

print('Train: \n')

train.shape

train.head()

explore(train)

print('Test: \n')

test.head()

explore(test)
# Converting col names to lowercase for ease

train.columns = [col.lower() for col in train.columns]

test.columns = [col.lower() for col in test.columns]

train.columns
# Solving null values in 'province_state' column

train['province_state'].fillna('All', inplace= True)

test['province_state'].fillna('All', inplace= True)

train.head()
#Feature extraction from 'date'

for df in [train, test]:

    df['date'] = pd.to_datetime(df['date'])

    df['day'] = df['date'].dt.day

    df['dayofweek'] = df['date'].dt.dayofweek

    df['dayofyear'] = df['date'].dt.dayofyear

    df['weekofyear'] = df['date'].dt.weekofyear

    df['month'] = df['date'].dt.month

    df['quarter'] = df['date'].dt.quarter

    

train.head()

test.head()
# Datatype conversions

train.dtypes

train.describe()
# Drop 'id'

train.drop(columns= 'id', inplace= True)



# Convert cat cols

cat_cols= ['province_state', 'country_region'] #'weekday'

train[cat_cols]= train[cat_cols].astype('category')

test[cat_cols]= test[cat_cols].astype('category')



# Convert numeric cols

train[['confirmedcases', 'fatalities']]= train[['confirmedcases', 'fatalities']].astype('int32')



train[['day', 'month', 'dayofweek', 'weekofyear', 'quarter']]= train[['day', 'month', 'dayofweek', 'weekofyear', 'quarter']].astype('int8')

test[['day', 'month', 'dayofweek', 'weekofyear', 'quarter']]= test[['day', 'month', 'dayofweek', 'weekofyear', 'quarter']].astype('int8')
def fillState(state, country):

    if state == 'All': return 'All ' + country

    return state



# Fill na state

for df in [train, test]:

    df['province_state'] = df.loc[:, ['province_state', 'country_region']].apply(lambda x : fillState(x['province_state'], x['country_region']), axis=1)

    df['province_state']= df['province_state'].astype('category')



train.head()

test.head()


train= train.drop(columns= 'date')

test= test.drop(columns= 'date')

train.head()

test.head()
import category_encoders as ce

meanEncCols= list(train.select_dtypes(include= 'category'))

ce_target1= ce.LeaveOneOutEncoder(cols = meanEncCols, sigma= 0.25) 

ce_target2= ce.LeaveOneOutEncoder(cols = meanEncCols, sigma= 0.25) 

ce_target1= ce_target1.fit(train.drop(columns= ['confirmedcases', 'fatalities']), train['confirmedcases'])

ce_target2= ce_target2.fit(train.drop(columns= ['confirmedcases', 'fatalities']), train['fatalities'])
submission = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in test.country_region.unique():

    data= train.loc[train.country_region== country, : ]

    test1= test.loc[test.country_region== country, : ]

    test2= test.loc[test.country_region== country, : ]    

    testForcastId= test1['forecastid']

    test1.drop(columns= 'forecastid', inplace= True)

    test2.drop(columns= 'forecastid', inplace= True)



    data1= data.drop(columns= ['confirmedcases', 'fatalities'])

    data2= data.drop(columns= ['confirmedcases', 'fatalities'])   

    y1_data= data['confirmedcases']

    y2_data= data['fatalities']

    

    data1= ce_target1.transform(data1)

    data2= ce_target2.transform(data2)

    

    test1= ce_target1.transform(test1)

    test2= ce_target2.transform(test2)



    #Model Building

    from xgboost import XGBRegressor

    XG1= XGBRegressor(n_estimators= 1000, n_jobs= -1)

    XG1.fit(data1, y1_data)   

    cc_pred_test = XG1.predict(test1)



    XG2= XGBRegressor(n_estimators= 1000, n_jobs= -1)

    XG2.fit(data2, y2_data)    

    f_pred_test = XG2.predict(test2)



    currentPreds = pd.DataFrame({'ForecastId': testForcastId, 'ConfirmedCases': cc_pred_test, 'Fatalities': f_pred_test})    

    submission = pd.concat([submission, currentPreds], axis= 0)

    

submission['ForecastId']= submission['ForecastId'].astype('int')
# Submission File

submission.to_csv('submission.csv', index=False)