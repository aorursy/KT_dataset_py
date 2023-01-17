import numpy as np
import pandas as pd

import time
import datetime

# import the data

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

index = test_df.columns
index = list(index[1:])

X_train, y_train = train_df[index], train_df['Category']
X_test = test_df[index]

dic_crime = {item : i for i, item in enumerate(set(y_train.values))}
y_train = pd.DataFrame(map(lambda f:dic_crime[f], y_train.values), columns=['Category'])

dic_day = {item : i for i, item in enumerate(set(X_train['DayOfWeek']))}
X_train['Days'] = X_train['DayOfWeek'].map(lambda f:dic_day[f])
X_test['Days'] = X_test['DayOfWeek'].map(lambda f:dic_day[f])

dic_distric = {item : i for i, item in enumerate(set(X_train['PdDistrict']))}
X_train['PdDistrict'] = X_train['PdDistrict'].map(lambda f:dic_distric[f])
X_test['PdDistrict'] = X_test['PdDistrict'].map(lambda f:dic_distric[f])

X_train.drop(['DayOfWeek'], axis = 1)
X_test.drop(['DayOfWeek'], axis = 1)

X_train['Dates'] = X_train['Dates'].map(lambda x: parse(x) - parse(X_train['Dates'][0]))


