# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv', index_col=0)
train_df.dtypes
test_df.dtypes
train_df['engine-type'].value_counts()
test_df['engine-type'].value_counts()
del train_df['normalized-losses']

del train_df['make']

del train_df['fuel-system']

del train_df['engine-type']
del test_df['normalized-losses']

del test_df['make']

del test_df['fuel-system']

del test_df['engine-type']
train_df = train_df.replace('?', np.NaN)
test_df = test_df.replace('?', np.NaN)
train_df = train_df.dropna(how='any')
dummy_columns = ['fuel-type', 'aspiration', 'engine-location', 'drive-wheels', 'body-style']
dummy_train = pd.get_dummies(train_df[dummy_columns], drop_first=True)
dummy_train
train_df = pd.merge(train_df, dummy_train, left_index=True, right_index=True)
for col in dummy_columns:

    del train_df[col]
train_df['num-of-doors'] = train_df['num-of-doors'].map({'two':2, 'four':4})

train_df['num-of-cylinders'] = train_df['num-of-cylinders'].map({'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12})
test_df['num-of-doors'] = test_df['num-of-doors'].map({'two':2, 'four':4})

test_df['num-of-cylinders'] = test_df['num-of-cylinders'].map({'two':2, 'four':4, 'five':5, 'six':6})
train_df['bore'] = pd.to_numeric(train_df['bore'], errors='raise')

train_df['stroke'] = pd.to_numeric(train_df['stroke'], errors='raise')

train_df['horsepower'] = pd.to_numeric(train_df['horsepower'], errors='raise')

train_df['peak-rpm'] = pd.to_numeric(train_df['peak-rpm'], errors='raise')

train_df['price'] = pd.to_numeric(train_df['price'], errors='raise')
test_df['bore'] = pd.to_numeric(test_df['bore'], errors='ignore')

test_df['stroke'] = pd.to_numeric(test_df['stroke'], errors='ignore')

test_df['horsepower'] = pd.to_numeric(test_df['horsepower'], errors='ignore')

test_df['peak-rpm'] = pd.to_numeric(test_df['peak-rpm'], errors='ignore')

test_df['price'] = pd.to_numeric(test_df['price'], errors='ignore')
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
test_df['horsepower'] = imputer.fit_transform(test_df['horsepower'].values.reshape(-1, 1))

test_df['price'] = imputer.fit_transform(test_df['price'].values.reshape(-1, 1))
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
test_df['bore'] = imputer.fit_transform(test_df['bore'].values.reshape(-1, 1))

test_df['stroke'] = imputer.fit_transform(test_df['stroke'].values.reshape(-1, 1))
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
test_df['num-of-doors'] = imputer.fit_transform(test_df['num-of-doors'].values.reshape(-1, 1))

test_df['peak-rpm'] = imputer.fit_transform(test_df['peak-rpm'].values.reshape(-1, 1))
dummy_test = pd.get_dummies(test_df[dummy_columns], drop_first=True)

test_df = pd.merge(test_df, dummy_test, left_index=True, right_index=True)

for col in dummy_columns:

    del test_df[col]
dummy_train
dummy_test
X_train = train_df.drop(['symboling'], axis=1).to_numpy()

y_train = train_df['symboling'].to_numpy()
train_df
X_test = test_df.to_numpy()
test_df
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

reg = DecisionTreeRegressor(criterion = 'mse')

params = {'max_depth':[ 2, 7, 10, 15, 20]}

gscv = GridSearchCV(reg, params, cv=136, scoring='neg_mean_squared_error')

gscv.fit(X_train, y_train)
print('%.3f  %r' % (gscv.best_score_, gscv.best_params_))
scores = gscv.cv_results_['mean_test_score']

params = gscv.cv_results_['params']

for score, param in zip(scores, params):

  print('%.3f  %r' % (score, param))
reg = DecisionTreeRegressor(criterion = 'mse', max_depth = 2)
reg.fit(X_train, y_train)
reg.score(X_train, y_train)
p_test = reg.predict(X_test)
p_test
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p_test

submit_df
submit_df.to_csv('submission2.csv')