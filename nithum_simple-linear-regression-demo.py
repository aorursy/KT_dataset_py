import numpy as np

import pandas as pd

import sklearn
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
train_data['logSalePrice'] = np.log(train_data['SalePrice'])
from sklearn.cross_validation import train_test_split
train, valid = train_test_split(train_data, test_size = 0.2, random_state = 10)
train.shape
valid.shape
pd.set_option('display.max_columns', 100)

train.head()
train.describe()
corr = train.select_dtypes(include = ['float64', 'int64']).corr()

corr['logSalePrice'].sort_values(ascending = False)
features = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath']

label = 'logSalePrice'
X_train = train[features]

y_train = train[label]



X_valid = valid[features]

y_valid = valid[label]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
house = X_valid.head(1)
house
model.predict(house)
np.exp(model.predict(house))[0]
valid['SalePrice'].head(1)
#house = pd.DataFrame(columns=features)

#house.loc[0] = [5,200,100,400,1]
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):

    return(mean_squared_error(y_true, y_pred)**0.5)
y_pred = model.predict(X_valid)
rmse(y_valid, y_pred)
X_all = train_data[features]

y_all = train_data[label]



X_test = test_data[features]

output_ids = test_data['Id']
model.fit(X_all, y_all)
y_output = model.predict(X_test)
X_test.describe()
X_test = X_test.fillna(X_all.mean())
X_test.describe()
y_output = model.predict(X_test)
y_output = np.exp(y_output)
output_df = pd.concat([output_ids, pd.Series(y_output, name = 'SalePrice')], axis=1)
output_df.head()
output_df.to_csv('kaggle_submission.csv', index=False)
# Another model

from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators= 25)
model_rf.fit(X_all, y_all)
y_output = np.exp(model_rf.predict(X_test))
output_df = pd.concat([output_ids, pd.Series(y_output, name = 'SalePrice')], axis = 1)
output_df.to_csv('rf_output.csv',index=False)