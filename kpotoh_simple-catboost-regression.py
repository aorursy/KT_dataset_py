import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")

%matplotlib inline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV

from scipy.sparse import csr_matrix, hstack

import lightgbm as lgb

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

%matplotlib inline
def rmse(estimator, X_test, y_test):

    """metrics for this competition"""

    return - np.sqrt(((estimator.predict(X_test) - y_test) ** 2).mean())



def rmse_log(estimator, X_test, y_test):

    """metrics for this competition"""

    return - np.sqrt(((np.exp(estimator.predict(X_test)) - np.exp(y_test)) ** 2).mean())
train_df = pd.read_csv('../input/train_features.csv')

test_df = pd.read_csv('../input/test_features.csv')

target = pd.read_csv('../input/train_target.csv', index_col='id')
train_df.shape, test_df.shape
train_df['DepHour'] = train_df['CRSDepTime'] // 100

train_df['DepHour'].replace(to_replace=24, value=0, inplace=True)



test_df['DepHour'] = test_df['CRSDepTime'] // 100

test_df['DepHour'].replace(to_replace=24, value=0, inplace=True)



train_df['ArrHour'] = train_df['CRSArrTime'] // 100

train_df['ArrHour'].replace(to_replace=24, value=0, inplace=True)



test_df['ArrHour'] = test_df['CRSArrTime'] // 100

test_df['ArrHour'].replace(to_replace=24, value=0, inplace=True)



test_df.drop(['CRSDepTime', 'CRSArrTime', 'Year', ], axis=1, inplace=True)

train_df.drop(['CRSDepTime', 'CRSArrTime', 'Year', ], axis=1, inplace=True)



train_df['target'] = target
train_df.head()
# Compute the correlation matrix

corr = train_df.iloc[:, [0,1,2,4,6, -4,-3,-2,-1]].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
test_df.drop(['CRSElapsedTime', ], axis=1, inplace=True)

train_df.drop(['CRSElapsedTime', ], axis=1, inplace=True)
train_df.head()
train_df.target.plot.hist(bins=50)
_ = plt.hist(np.log(train_df.target), bins=50)
train_df.groupby('FlightNum').target.mean().sort_values().plot(style='*') #.plot.bar()
train_df.groupby('FlightNum').target.size().sort_values().plot(style='+')
train_df.FlightNum.nunique(), train_df.TailNum.nunique()
X_train = train_df[['Distance', 'DepHour', 'FlightNum']].values

X_test = test_df[['Distance', 'DepHour', 'FlightNum']].values

y_train_log = np.log(train_df.target.values)

y_train = train_df.target.values

# X_train_part, X_valid, y_train_part, y_valid = \

#     train_test_split(X_train, y_train, 

#                      test_size=0.3, random_state=17)
linear_pipe = Pipeline([('scaler', StandardScaler()),

                       ('linear', LinearRegression())])



ert = ExtraTreesRegressor(criterion='mae', n_estimators=30, )

lcv = cross_val_score(linear_pipe, X_train, y_train, scoring=rmse , cv=5)

print(-lcv.mean(), lcv.std())
linear_pipe.fit(X_train, y_train)

linear_test_pred = linear_pipe.predict(X_test)
pd.Series(linear_test_pred, name='DelayTime').to_csv('linear_2feat.csv', 

                                           index_label='id', header=True)
cbr = CatBoostRegressor(logging_level='Silent', random_state=45, 

                        early_stopping_rounds=300, )
# There are a little number (~5) of None in categorial column, fill it!

train_df.fillna(0, inplace=True)

test_df.fillna(0, inplace=True)
train_df.head()
# simple catboost regressor without setup

cbr.fit(train_df.drop('target', axis=1), target, cat_features=[0,1,2,3,5,6,7], plot=True)
cb_pred = cbr.predict(test_df, verbose=True)
pd.Series(cb_pred, name='DelayTime').to_csv('simple_CBR.csv', 

                                           index_label='id', header=True)
_ = plt.hist(cb_pred, bins=50)
zeros_submit_score = 68.79140

cb_pred_modif = cb_pred + zeros_submit_score - np.sqrt(np.square(target.values).mean())

pd.Series(cb_pred_modif, name='DelayTime').to_csv('simple_CBR_modif.csv', 

                                           index_label='id', header=True)