import numpy as np

import pandas as pd 

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_log_error
train_data = pd.read_csv('../input/house-price-prediction-challenge/train.csv')

test_data = pd.read_csv('../input/house-price-prediction-challenge/test.csv')
train_data.head()
test_data.head()
lr= LinearRegression()
x_train = train_data.drop(['POSTED_BY','BHK_OR_RK','ADDRESS','LONGITUDE','LATITUDE','TARGET(PRICE_IN_LACS)'], axis=1)

y_train = train_data['TARGET(PRICE_IN_LACS)']



lr.fit(x_train,y_train)
lr.coef_
x_test = test_data.drop(['POSTED_BY','BHK_OR_RK','ADDRESS','LONGITUDE','LATITUDE'], axis=1)

y_test = pd.read_csv('../input/house-price-prediction-challenge/sample_submission.csv')



y_pred = lr.predict(x_test)
err = np.sqrt(mean_squared_log_error(y_test, abs(y_pred)))
err