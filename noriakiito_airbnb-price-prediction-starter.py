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
train_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/train.csv', index_col=0)
train_df
test_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/test.csv', index_col=0)
test_df
df = pd.concat([train_df.drop('price', axis=1), test_df])
df
df = pd.concat([df, pd.get_dummies(df['neighbourhood_group'])], axis=1)
df = df.drop('neighbourhood_group', axis=1)
df
print(df.isnull().sum())
df = df.drop('neighbourhood', axis=1)
df
df = pd.concat([df, pd.get_dummies(df['room_type'])], axis=1)
df = df.drop('room_type', axis=1)
df
df = df.drop(['name', 'host_id', 'host_name', 'number_of_reviews', 'Bronx',
              'Queens', 'Staten Island', 'Shared room', 'Private room'], axis=1)
df
nrow, ncol = train_df.shape
price_df = train_df[['price']]
train_df = df[:nrow]
train_df = pd.concat([train_df, price_df], axis=1)
train_df
nrow, ncol = train_df.shape
test_df = df[nrow:]
test_df
temp = train_df[['longitude']]
train_df = train_df[np.abs(train_df-train_df.mean()) <= 2. * (train_df.std())]
train_df[['longitude']] = temp
train_df = train_df.dropna()
train_df
test_df = test_df.fillna(test_df.mode().iloc[0])
test_df
import matplotlib.pyplot as plt
train_df.hist(bins=20, figsize=(20,15))
# creates a figure with 10 (width) x 5 (height) inches
plt.rcParams['figure.figsize'] = [10, 5]
plt.show()
import matplotlib.pyplot as plt
test_df.hist(bins=20, figsize=(20,15))
# creates a figure with 10 (width) x 5 (height) inches
plt.rcParams['figure.figsize'] = [10, 5]
plt.show()
train_df.min()
train_df['calculated_host_listings_count'] = train_df['calculated_host_listings_count'].apply(np.log)
import matplotlib.pyplot as plt
train_df.hist(bins=20, figsize=(20,15))
# creates a figure with 10 (width) x 5 (height) inches
plt.rcParams['figure.figsize'] = [10, 5]
plt.show()
test_df.min()
test_df['calculated_host_listings_count'] = test_df['calculated_host_listings_count'].apply(np.log)
import matplotlib.pyplot as plt
test_df.hist(bins=20, figsize=(20,15))
# creates a figure with 10 (width) x 5 (height) inches
plt.rcParams['figure.figsize'] = [10, 5]
plt.show()
import seaborn as sns
corrmat = train_df.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.set(font_scale=1.2)
hm = sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10})
plt.show()
import seaborn as sns
corrmat = test_df.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.set(font_scale=1.2)
hm = sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10})
plt.show()
X = train_df.drop(['price'], axis=1).to_numpy()
y = train_df['price'].to_numpy()
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

print("Parameter optimization")
xgb_model = xgb.XGBRegressor()
reg_xgb = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
reg_xgb.fit(X, y)
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(X.shape[1], input_dim=X.shape[1], kernel_initializer='normal', activation='swish'))
    model.add(Dense(16, kernel_initializer='normal', activation='swish'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)
# define the grid search parameters
optimizer = ['Adam']
batch_size = [4]
epochs = [20]
param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)
reg_dl = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
reg_dl.fit(X, y)
from sklearn.svm import SVR

reg_svr = SVR(kernel='rbf', gamma=0.1)
reg_svr.fit(X, y)
# second feature matrix
X2 = pd.DataFrame( {'XGB': reg_xgb.predict(X),
     'DL': reg_dl.predict(X).ravel(),
     'SVR': reg_svr.predict(X),
    })
X_test = test_df.to_numpy()
# second-feature modeling using linear regression
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(X2, y)

# prediction using the test set
X_test2 = pd.DataFrame( {'XGB': reg_xgb.predict(X_test),
     'DL': reg_dl.predict(X_test).ravel(),
     'SVR': reg_svr.predict(X_test),
    })

p = reg.predict(X_test2)
submit_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/sampleSubmission.csv')
submit_df['price'] = p
submit_df.to_csv('submission2.csv', index=False)

