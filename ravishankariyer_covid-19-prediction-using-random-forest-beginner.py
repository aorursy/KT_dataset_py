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

submit = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')
x_features = train_data 

x_features = x_features.drop(columns=['Province_State','ConfirmedCases','Fatalities'])

x_features.info()
x_features.Date = pd.to_datetime(x_features.Date)

x_features.Date = x_features.Date.astype(int)

x_features.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

x_features.Country_Region = le.fit_transform(x_features.Country_Region)

x_features.info()

x_features.head(200)
y_target_con = train_data.ConfirmedCases

y_target_con.head()
test_features = test_data.drop(columns=['Province_State'])

test_features.Date = pd.to_datetime(test_features.Date)

test_features.Date = test_features.Date.astype(int)

test_features.Country_Region = le.fit_transform(test_features.Country_Region)

test_features.info()

test_features.head(200)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,random_state=10)

rf.fit(x_features,y_target_con)
predict_con = rf.predict(test_features)



predict_con
y_target_fat = train_data.Fatalities

y_target_fat.head()
rf.fit(x_features,y_target_fat)
predict_fat = rf.predict(test_features)



predict_fat
predict_fat[0:100]
submit.ForecastId = test_data.ForecastId

submit.ConfirmedCases = predict_con

submit.Fatalities = predict_fat



submit.head(25)
submit.to_csv('submission.csv',index=False)