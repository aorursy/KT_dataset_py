import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')
test_data = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
submit = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')
x_tr = train_data 
x_tr = x_tr.drop(columns=['Province_State','ConfirmedCases','Fatalities'])
x_tr.info()
x_tr.Date = pd.to_datetime(x_tr.Date)
x_tr.Date = x_tr.Date.astype(int)
x_tr.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x_tr.Country_Region = le.fit_transform(x_tr.Country_Region)
x_tr.head(200)
y_target = train_data.ConfirmedCases
y_target.head()
test_features = test_data.drop(columns=['Province_State'])
test_features.Date = pd.to_datetime(test_features.Date)
test_features.Date = test_features.Date.astype(int)
test_features.Country_Region = le.fit_transform(test_features.Country_Region)
test_features.info()
test_features.head(200)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state=10)
rf.fit(x_tr,y_target)
predict = rf.predict(test_features)

predict
y_target_fat = train_data.Fatalities
y_target_fat.head()
rf.fit(x_tr,y_target_fat)
predict_fat = rf.predict(test_features)

predict_fat
predict_fat[0:100]
submit.ForecastId = test_data.ForecastId
submit.ConfirmedCases = predict
submit.Fatalities = predict_fat

submit.head(25)
submit.to_csv('submission.csv',index=False)
