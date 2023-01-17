import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from sklearn.linear_model import LinearRegression, SGDRegressor

from sklearn import metrics

import lightgbm as lgb

from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.
train = pd.read_csv('../input/champs-scalar-coupling/train.csv')

test = pd.read_csv('../input/champs-scalar-coupling/test.csv')

structures = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
print('Train shape: {}'.format(train.shape))

print('Test shape: {}'.format(test.shape))
train.info()
test.info()
train.describe()
train[['molecule_name','scalar_coupling_constant']].groupby('molecule_name').mean()[:100]
train[['type','scalar_coupling_constant']].groupby('type').count()
sns.distplot(train['scalar_coupling_constant'])
train = pd.merge(train, structures, how = 'left',left_on  = ['molecule_name', 'atom_index_0'],

right_on = ['molecule_name',  'atom_index'])



train = pd.merge(train, structures, how = 'left',left_on  = ['molecule_name', 'atom_index_1'],

right_on = ['molecule_name',  'atom_index'])
test = pd.merge(test, structures, how = 'left',left_on  = ['molecule_name', 'atom_index_0'],

right_on = ['molecule_name',  'atom_index'])



test = pd.merge(test, structures, how = 'left',left_on  = ['molecule_name', 'atom_index_1'],

right_on = ['molecule_name',  'atom_index'])
train.head()
train['dist'] = ((train['x_y'] - train['x_x'])**2 + (train['y_y'] - train['y_x'])**2 + 

(train['z_y'] - train['z_x'])**2 ) ** 0.5



test['dist'] = ((test['x_y'] - test['x_x'])**2 + (test['y_y'] - test['y_x'])**2 +

(test['z_y'] - test['z_x'])**2 ) ** 0.5
train.head()
train.columns
features = ['atom_index_x', 'x_x', 'y_x','z_x', 'atom_index_y', 'x_y', 'y_y', 'z_y', 'dist']
X_train,X_val,y_train,y_val = train_test_split(train[features], train['scalar_coupling_constant'],test_size=0.2)
xgb = XGBRegressor()

xgb.fit(X_train,y_train)

preds = xgb.predict(X_val)
np.log(metrics.mean_absolute_error(y_val,preds))
test_predictions = xgb.predict(test[features])
sns.distplot(test_predictions)
submission = pd.DataFrame()

submission['id'] = test['id']

submission['scalar_coupling_constant'] = test_predictions
submission.to_csv('usingXGBoost.csv.gz',index=False,compression='gzip')