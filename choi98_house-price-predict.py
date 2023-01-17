import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col = 'Id')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
data.head()
#plt.figure(figsize = (20, 8))

#sns.lmplot(x = 'YrSold', y = 'SalePrice', data = data)

#sns.swarmplot(x=data['Fence'], y=data['SalePrice'])
y = data['SalePrice']

features = ['LotArea', 'OverallQual', 'GarageCars', 'YearBuilt', 'YearRemodAdd', 'BedroomAbvGr', 'FullBath', 'GrLivArea', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea']

X = data[features]

test_X = test[features]

test_X = test_X.fillna(value=test_X.GarageCars.mean())

test_X = test_X.fillna(value=test_X.GarageArea.mean())

X = X.dropna(axis=0)

test_X.isnull().sum()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor()

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y)

model.fit(train_X, train_y)

preds = model.predict(val_X)

from sklearn.metrics import mean_absolute_error

err = mean_absolute_error(val_y, preds)

preds = model.predict(test_X)
submission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')

submission['SalePrice'] = preds
submission.to_csv('./submission.csv', index=False)