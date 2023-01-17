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
import pandas as pd

train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
train.head()

import numpy as np

train.fillna(' ',inplace=True)

train['Region']=train['Province_State']+train['Country_Region']

train.drop('Province_State',axis=1,inplace=True)

train.drop('Country_Region',axis=1,inplace=True)

train.head()
region_encoded = dict(enumerate(train['Region'].unique()))

region_encoded = dict(map(reversed, region_encoded.items()))

train['Region_Code'] = train['Region'].apply(lambda x: region_encoded[x])

train.head()
from datetime import datetime

import time

train['Mon'] = train['Date'].apply(lambda x: int(x.split('-')[1]))

train['Day'] = train['Date'].apply(lambda x: int(x.split('-')[2]))
train_region=train

train.drop('Region',axis=1,inplace=True)
train['serial'] = train['Mon'] * 30 + train['Day']
train['serial'] = train['serial'] - train['serial'].min()

train.head()
train_date=train

train.drop('Date',axis=1,inplace=True)
numeric_features_X = ['Region_Code','Mon','Day']

numeric_features_Y = ['ConfirmedCases', 'Fatalities']

train_numeric_X = train[numeric_features_X]

train_numeric_Y = train[numeric_features_Y]

# train

test.fillna(' ',inplace=True)

test['Region']=test['Province_State']+test['Country_Region']

test.drop('Province_State',axis=1,inplace=True)

test.drop('Country_Region',axis=1,inplace=True)

test
test['Region_Code'] = test['Region'].apply(lambda x: region_encoded[x] if x in region_encoded else max(region_encoded.values())+1)
test_region=test

test.drop('Region',axis=1,inplace=True)
test['Mon'] = test['Date'].apply(lambda x: int(x.split('-')[1]))

test['Day'] = test['Date'].apply(lambda x: int(x.split('-')[2]))

test['serial'] = test['Mon'] * 30 + test['Day']

test['serial'] = test['serial'] - test['serial'].min()

test
test_date=test

test.drop('Date',axis=1,inplace=True)
test_numeric_X = test[numeric_features_X]

test_numeric_X.isnull().sum()
# from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler

# from sklearn.linear_model import LinearRegression
# pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])

# pipeline.fit(train_numeric_X, train_numeric_Y)

# predicted = pipeline.predict(test_numeric_X)
# submission1=submission1.astype(np.int32)
# submission1
# df = pd.DataFrame(data=submission1, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('submission.csv', index=False)
from sklearn.ensemble import RandomForestClassifier

# pipeline = Pipeline([('scaler', StandardScaler()), ('rfc', RandomForestClassifier(random_state=1,n_estimators=10,n_jobs=4,max_depth=5))])

# pipeline.fit(train_numeric_X, train_numeric_Y)

my_model=RandomForestClassifier(n_estimators=30)

my_model.fit(train_numeric_X, train_numeric_Y)

predicted = my_model.predict(test_numeric_X)
# from xgboost import XGBRegressor

# train_numeric_y1=train_numeric_Y['ConfirmedCases']

# train_numeric_y2=train_numeric_Y['Fatalities']

# my_model1=XGBRegressor(random_state=1, n_estimators=1000, learning_rate=0.05)

# my_model2=XGBRegressor(random_state=1, n_estimators=1000, learning_rate=0.05)

# my_model1.fit(train_numeric_X, train_numeric_y1)

# predicted1=my_model1.predict(test_numeric_X)

# my_model2.fit(train_numeric_X, train_numeric_y2)

# predicted2=my_model2.predict(test_numeric_X)

# from sklearn.ensemble import AdaBoostClassifier

# adaboost_model_for_ConfirmedCases = AdaBoostClassifier(n_estimators=5)

# adaboost_model_for_ConfirmedCases.fit(train_numeric_X, train_numeric_Y[numeric_features_Y[0]])

# adaboost_model_for_Fatalities = AdaBoostClassifier(n_estimators=5)

# adaboost_model_for_Fatalities.fit(train_numeric_X, train_numeric_Y[numeric_features_Y[1]])

# predicted = adaboost_model_for_ConfirmedCases.predict(test_numeric_X)

# predicted2 = adaboost_model_for_Fatalities.predict(test_numeric_X)



submission1 = np.vstack((test['ForecastId'], predicted[:,0],predicted[:,1])).T

# submission1 = np.vstack((test['ForecastId'], predicted1,predicted2)).T

submission1 = submission1.astype(np.int32)
df = pd.DataFrame(data=submission1, columns=['ForecastId','ConfirmedCases','Fatalities'])

df.to_csv('submission.csv', index=False)