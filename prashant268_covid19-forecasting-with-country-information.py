import pandas as pd

import numpy as np
cleaned_train = pd.read_csv('../input/covid-clean/clean_train.csv')

cleaned_test = pd.read_csv('../input/covid-clean/clean_test.csv')

submission = pd.read_csv('../input/covid-clean/submission.csv')
cleaned_train.columns
X = cleaned_train[['day_from_jan_first','Lat','Long',

                   'medianage','urbanpop','hospibed',

                   'lung','avgtemp','avghumidity','days_from_firstcase']]
X_fat = cleaned_train[['day_from_jan_first','Lat','Long',

                   'medianage','urbanpop','hospibed',

                   'lung','avgtemp','avghumidity','days_from_firstcase','ConfirmedCases']]
y1 = cleaned_train[['ConfirmedCases']]

y2 = cleaned_train[['Fatalities']]
from sklearn.tree import DecisionTreeRegressor

regressor_1=DecisionTreeRegressor(max_depth=30,max_features=8,min_samples_split=2,min_samples_leaf=1)

regressor_2=DecisionTreeRegressor(max_depth=30,max_features=8,min_samples_split=2,min_samples_leaf=1)

regressor_1.fit(X,y1)

regressor_2.fit(X_fat,y2)
test_X = cleaned_test[['day_from_jan_first','Lat','Long',

                   'medianage','urbanpop','hospibed',

                   'lung','avgtemp','avghumidity','days_from_firstcase']]

test_X_fat  = cleaned_test[['day_from_jan_first','Lat','Long',

                   'medianage','urbanpop','hospibed',

                   'lung','avgtemp','avghumidity','days_from_firstcase']]
y_conf = regressor_1.predict(test_X)

y_conf = np.where(y_conf<0,0,np.rint(y_conf))
test_X_fat['ConfirmedCases'] = y_conf
y_fat = regressor_2.predict(test_X_fat)

y_fat = np.where(y_fat<0,0,np.rint(y_fat))
submission=pd.DataFrame(columns=submission.columns)
submission['ForecastId'] = cleaned_test['ForecastId']

submission['ConfirmedCases'] = y_conf

submission['Fatalities'] = y_fat
submission.to_csv('submission.csv',index=False)