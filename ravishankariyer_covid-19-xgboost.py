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

import numpy as np
train_data = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test_data = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X_features = train_data

X_features.info()
X_features = X_features.drop(columns=['Province_State','ConfirmedCases','Fatalities','Id'])

X_features.Country_Region = le.fit_transform(X_features.Country_Region)

X_features.Date = pd.to_datetime(X_features.Date)

X_features.Date = X_features.Date.astype(int)

X_features.info()

X_features.head(20)
y_target_con = train_data

y_target_con = y_target_con.drop(columns=['Id','Date','Country_Region','Province_State','Fatalities'])

y_target_con.info()

y_target_con.head()
test_features = test_data

test_features.Country_Region = le.fit_transform(test_features.Country_Region)

test_features.Date = pd.to_datetime(test_features.Date)

test_features.Date.astype(int)

test_features.drop(columns=['Province_State'])

test_features.info()

test_features.head()
test_features = test_features.drop(columns=['ForecastId','Province_State'],axis=1)

test_features.info()

test_features.head()
test_features.Date = test_features.Date.astype(int)

test_features.info()
from xgboost import XGBRegressor

model_con1 = XGBRegressor()

con_target = train_data.ConfirmedCases

model_con1.fit(X_features,con_target)
predict_con= model_con1.predict(test_features)

predict_con
fat = train_data.Fatalities

fat
model_fat1 = XGBRegressor()

model_fat1.fit(X_features,fat)
predict_fat = model_fat1.predict(test_features)

predict_fat
FinalSubmit = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')

FinalSubmit.ConfirmedCases = predict_con

FinalSubmit.Fatalities = predict_fat

FinalSubmit.head(20)
FinalSubmit.describe()
FinalSubmit.to_csv('submission.csv',index=False)
FinalSubmit.to_csv('sample_submission.csv',index=False)