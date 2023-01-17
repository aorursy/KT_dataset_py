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
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
print(train.shape)

print(test.shape)

print(submission.shape)
train.head()
test.head()
submission.head()
train.info()
test.info()
X_train=train.drop(columns=['Id','ConfirmedCases','Fatalities','Date'])

y_train_cc=train.ConfirmedCases

y_train_ft=train.Fatalities
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

impute=SimpleImputer(strategy='most_frequent')

X_train_1=impute.fit_transform(X_train)

X_train_2=OneHotEncoder().fit_transform(X_train_1)
X_test=test.drop(columns=['ForecastId','Date'])

X_test_1=impute.fit_transform(X_test)

X_test_2=OneHotEncoder().fit_transform(X_test_1)
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

model_cc=RandomForestRegressor()

model_cc.fit(X_train_2, y_train_cc)

model_cc.score(X_train_2, y_train_cc)
y_pred_cc=model_cc.predict(X_test_2)
model_ft=RandomForestRegressor()

model_ft.fit(X_train_2,y_train_ft)

model_ft.score(X_train_2, y_train_ft)
y_pred_ft=model_ft.predict(X_test_2)

y_pred_ft
submission.head()
result=pd.DataFrame({'ForecastId':submission.ForecastId, 'ConfirmedCases':y_pred_cc, 'Fatalities':y_pred_ft})

result.to_csv('/kaggle/working/submission.csv', index=False)

data=pd.read_csv('/kaggle/working/submission.csv')

data.head()