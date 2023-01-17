import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from math import sqrt



from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2, RFE

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
data = pd.read_csv('../input/bits-f464-l1/train.csv')

test_data = pd.read_csv('../input/bits-f464-l1/test.csv')
data.head()
train_ID = data['id']

test_ID = test_data['id']

data.drop(['id'], axis=1, inplace=True)

test_data.drop(['id'], axis=1, inplace=True)

data.shape, test_data.shape
features = data.drop(['label'], axis=1)

labels = data['label'].reset_index(drop=True)

test_features = test_data



# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset

all_features = pd.concat([features, test_features]).reset_index(drop=True)

print(all_features.shape)

all_features.head()
all_features = (all_features - all_features.mean())/all_features.std()

all_features.dropna(axis=1, inplace=True)

all_features.head()
X = all_features.iloc[:len(labels), :]

X_test = all_features.iloc[len(labels):, :]

X.shape, labels.shape, X_test.shape
estimator = LinearRegression()

selector = RFE(estimator, 80, step=1, verbose=True)

selector = selector.fit(X, labels)
X = selector.transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=4000, learning_rate=0.01, max_depth=4, max_features='sqrt', loss='huber', verbose=1)

model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print(sqrt(mean_squared_error(y_val, y_pred)), r2_score(y_val, y_pred))
test_data.head()
X_test = selector.transform(X_test)

print(X_test.shape)

y_res = model.predict(X_test)

y_res
res_df = pd.DataFrame(data = {'id': test_ID, 'label' : y_res})
res_df.head()
res_df.to_csv('res_4.csv', index=False)