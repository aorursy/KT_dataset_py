import pandas as pd

import numpy as np

from lightgbm import LGBMRegressor
training_set = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv', index_col = 'Id')

test_set = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv', index_col = 'ForecastId')
training_set.head()
test_set.head()
training_set.describe()
test_set.describe()
training_set_2 = training_set.copy()

test_set_2 = test_set.copy()
train_confirmedcases = np.array(training_set['ConfirmedCases'].astype(int))

train_fatalities = np.array(training_set['Fatalities'].astype(int))
train_confirmedcases[0:5]
train_fatalities[0:5]
dataset = pd.concat([training_set.drop(['ConfirmedCases', 'Fatalities'], axis=1), test_set])

dataset.count()
dataset['Province_State'].fillna('No_Data', inplace = True)

dataset.head()
dataset = pd.get_dummies(dataset, columns = dataset.columns)

dataset.head()
split = training_set.shape[0]

X_train = dataset[:split]

X_test= dataset[split:]
X_train.count()
X_test.count()
regressor = LGBMRegressor(boosting_type = 'dart', learning_rate= 0.001, random_state = 0, n_jobs = -1)

regressor.fit(X_train.values, train_confirmedcases)

predicted_confirmedcases = regressor.predict(X_test.values)

predicted_confirmedcases
regressor_2 = LGBMRegressor(boosting_type = 'dart', learning_rate= 0.0001, num_leaves = 124, random_state = 0, n_jobs = -1)

regressor_2.fit(X_train.values, train_fatalities)

predicted_fatalities = regressor_2.predict(X_test.values)

predicted_fatalities
submission = pd.DataFrame({'ForecastId': test_set.index,'ConfirmedCases': predicted_confirmedcases,'Fatalities': predicted_fatalities})

submission.to_csv('submission.csv',index = False)