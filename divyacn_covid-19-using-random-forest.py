import numpy as np 

import pandas as pd 

import pandas as pd

import numpy as np
train_data = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test_data = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
features = train_data 

features = features.drop(columns=['Province_State','ConfirmedCases','Fatalities'])
features.Date = pd.to_datetime(features.Date)

features.Date = features.Date.astype(int)

features.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

features.Country_Region = le.fit_transform(features.Country_Region)

features.info()

features.head(200)
target_con = train_data.ConfirmedCases

target_con.head()
test_features = test_data.drop(columns=['Province_State'])

test_features.Date = pd.to_datetime(test_features.Date)

test_features.Date = test_features.Date.astype(int)

test_features.Country_Region = le.fit_transform(test_features.Country_Region)

test_features.info()

test_features.head(200)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,random_state=10)

rf.fit(features,target_con)
predict = rf.predict(test_features)



predict
target_fat = train_data.Fatalities

target_fat.head()
rf.fit(features,target_fat)
predict_fat = rf.predict(test_features)



predict_fat
submit = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')

submit.ForecastId = test_data.ForecastId

submit.ConfirmedCases = predict

submit.Fatalities = predict



submit.head(25)
submit.to_csv('submission.csv',index=False)