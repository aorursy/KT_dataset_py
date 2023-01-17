# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime



from scipy.stats import zscore, boxcox

from sklearn.preprocessing import QuantileTransformer, LabelEncoder, MinMaxScaler

from sklearn.model_selection import RepeatedKFold, cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.model_selection import RepeatedKFold, cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error, make_scorer

from sklearn.ensemble import VotingRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



## NN ##

from keras.models import Sequential

from keras.layers import Dense as Dense2

from keras.wrappers.scikit_learn import KerasClassifier

from keras import regularizers

from keras import callbacks



from tensorflow.keras.layers import Dense, Flatten, Conv2D

from tensorflow.keras import Model

from tensorflow import keras

from tensorflow.keras import layers

import tensorflow as tf

#######





sns.set_style('darkgrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.options.mode.chained_assignment = None
pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv').head()
X_train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv')

stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')

features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv')

X_train = X_train.merge(stores, on='Store').merge(features.drop('IsHoliday', axis=1), on=['Store', 'Date'])

X_train.fillna(0, inplace=True) ## only NaN columns are the markdowns. Set to 0 if NaN ##

X_train['Date'] = pd.to_datetime(X_train['Date'])

# X_train_no_neg = X_train[X_train['Weekly_Sales'] >= 0].reset_index(drop=True)



# y_train = X_train[['Store', 'Dept', 'Date', 'Weekly_Sales']]

# X_train.drop('Weekly_Sales', axis=1, inplace=True)



# y_train_no_neg = X_train_no_neg[['Store', 'Dept', 'Date', 'Weekly_Sales']]

# X_train_no_neg.drop('Weekly_Sales', axis=1, inplace=True)



# print(X_train.shape, X_train_no_neg.shape, (X_train.shape[0] - X_train_no_neg.shape[0]) / X_train.shape[0])



X_train.head()
X_train['Date'].dtype
X_test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv')

X_test = X_test.merge(stores, on='Store').merge(features.drop('IsHoliday', axis=1), on=['Store', 'Date'])

X_test.fillna(0, inplace=True) ## only NaN columns are the markdowns. Set to 0 if NaN ##



X_test.head()
X_test['Store'].nunique(), X_test['Dept'].nunique(), X_test['Date'].nunique()
X_test.shape, X_train.shape
X_train['Year'] = pd.DatetimeIndex(X_train['Date']).year

X_train['Month'] = pd.DatetimeIndex(X_train['Date']).month

X_train['woy'] = pd.DatetimeIndex(X_train['Date']).weekofyear

X_train['quarter'] = pd.DatetimeIndex(X_train['Date']).quarter



X_test['Year'] = pd.DatetimeIndex(X_test['Date']).year

X_test['Month'] = pd.DatetimeIndex(X_test['Date']).month

X_test['woy'] = pd.DatetimeIndex(X_test['Date']).weekofyear

X_test['quarter'] = pd.DatetimeIndex(X_test['Date']).quarter



## for future reference ##

# df['dow'] = df.index.dayofweek

# df['doy'] = df.index.dayofyear
X_all['Store'].unique()
# ## Add 1-year `Weekly_Sales` lag ##



# X_all = pd.concat([X_train, X_test])

# X_all['Date2'] = pd.to_datetime(X_all['Date'], utc = True)

# X_all['52_Week_Lag'] = X_all['Date2'] - np.timedelta64(52,'W')

# X_all_temp = X_all[['Weekly_Sales', 'Date2', 'Store', 'Dept']]



# X_all = X_all.merge(X_all_temp,

#                     left_on=['Store', 'Dept', '52_Week_Lag'], 

#                     right_on=['Store', 'Dept', 'Date2'],

#                     how='inner',

#                     suffixes=('', '_y'))

# X_all.rename(columns={'Weekly_Sales_y': 'Weekly_Sales_Lag_52_Weeks'}, inplace=True)

# X_all = X_all[[col for col in X_all.columns if not col.endswith('_y')]]



# drop_cols = ['Date2_y', '1_Year_Lag']

# X_all.drop(['52_Week_Lag'], axis=1, inplace=True)



# X_all.isna().sum()
# X_train['Date2'] = pd.to_datetime(X_train['Date'], utc = True)

# X_test['Date2'] = pd.to_datetime(X_test['Date'], utc = True)



# X_train['Weekly_Sales_Lag_52_Weeks'] = X_train.merge(X_all, 

#                                                    left_on=['Store', 'Dept', 'Date2'], 

#                                                    right_on=['Store', 'Dept', 'Date2'],

#                                                    how='inner')['Weekly_Sales_Lag_52_Weeks']

# X_test['Weekly_Sales_Lag_52_Weeks'] = X_test.merge(X_all, 

#                                                  left_on=['Store', 'Dept', 'Date2'], 

#                                                  right_on=['Store', 'Dept', 'Date2'],

#                                                  how='inner')['Weekly_Sales_Lag_52_Weeks']



# X_test.head()
cols_num = [col for col in X_train.columns if X_train[col].dtype in [float, int]]

ncols = len(cols_num) // 4

fig, axes = plt.subplots(ncols=ncols, nrows=5, figsize=(30,16))



i = 1

for j, col in enumerate(cols_num):

    sns.distplot(X_train[col], bins=10, ax=axes[i-1][j % ncols])



    if j % ncols == (ncols - 1):

        i += 1

        

plt.tight_layout()
# sns.set(style="ticks", color_codes=True)



# for col in X_train.columns.drop('Weekly_Sales'):

#     sns.pairplot(data=X_train,

#                  y_vars=['Weekly_Sales'],

#     #              x_vars=['Weekly_Sales_Lag_1_Year', 'quarter', 'woy'],

#                  x_vars=col,

#                  hue='Year')
X_train['outlier'] = np.where((zscore(X_train['Weekly_Sales']) <= -2.5) | (zscore(X_train['Weekly_Sales']) >= 2.5), 1, 0)

num_outliers = X_train[X_train['outlier'] == 1]['Weekly_Sales'].count()



print('Number of `Weekly_Sales` outliers: {}\nPercent outliers: {:.2f}%'.format(num_outliers, num_outliers / X_train.shape[0] * 100))
num_neg_sales = X_train[X_train['Weekly_Sales'] < 0].shape[0]

print('Number of negative `Weekly_Sales` : {}\nPercent: {:.2f}%'.format(num_neg_sales, num_neg_sales / X_train.shape[0] * 100))



X_train['Weekly_Sales'] = np.where(X_train['Weekly_Sales'] < 0, 0, X_train['Weekly_Sales'])
X_train['Weekly_Sales_Log'] = np.log1p(X_train['Weekly_Sales'])

X_train['outlier'] = np.where((zscore(X_train['Weekly_Sales_Log']) <= -2.5) | (zscore(X_train['Weekly_Sales_Log']) >= 2.5), 1, 0)

num_outliers = X_train[X_train['outlier'] == 1]['Weekly_Sales_Log'].count()



print('Number of `Weekly_Sales_Log` outliers: {}\nPercent outliers: {:.2f}%'.format(num_outliers, num_outliers / X_train.shape[0] * 100))
## QuantileTransformer ##

## https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html ##

qt = QuantileTransformer(output_distribution='normal')

# y_train_qt = y_train.copy()

X_train['Weekly_Sales_tf'] = qt.fit_transform(np.array(X_train['Weekly_Sales'] + 1).reshape(-1, 1))

X_train['outlier'] = np.where((zscore(X_train['Weekly_Sales_tf']) <= -2.5) | (zscore(X_train['Weekly_Sales_tf']) >= 2.5), 1, 0)

num_outliers = X_train[X_train['outlier'] == 1]['Store'].count()



X_train_tf = X_train[X_train['outlier'] != 1]



print('QuantileTransformer - Number of `Weekly_Sales` outliers: {}\nPercent outliers: {:.2f}%'.format(num_outliers, num_outliers / X_train.shape[0] * 100))
## Add 1-year `Weekly_Sales` lag ##



X_all = pd.concat([X_train, X_test])

X_all['Date2'] = pd.to_datetime(X_all['Date'], utc = True)

X_all['52_Week_Lag'] = X_all['Date2'] - np.timedelta64(52,'W')

X_all_temp = X_all[['Weekly_Sales_tf', 'Date2', 'Store', 'Dept']]



X_all = X_all.merge(X_all_temp,

                    left_on=['Store', 'Dept', '52_Week_Lag'], 

                    right_on=['Store', 'Dept', 'Date2'],

                    how='left',

                    suffixes=('', '_y'))

X_all.rename(columns={'Weekly_Sales_tf_y': 'Weekly_Sales_tf_Lag_52_Weeks'}, inplace=True)

X_all = X_all[[col for col in X_all.columns if not col.endswith('_y')]]



## bad solution, fix later ##



## first try to fill missing 52-week lagged values (2,041 ~ 2%) with approximate year-lag values ##

for i in [53, 51, 54, 50, 55, 49, 56, 48]:

    X_all['n_Week_Lag'] = X_all['Date2'] - np.timedelta64(i,'W')

    X_all = X_all.merge(X_all_temp,

                    left_on=['Store', 'Dept', 'n_Week_Lag'], 

                    right_on=['Store', 'Dept', 'Date2'],

                    how='left',

                    suffixes=('', '_y'))

    X_all['Weekly_Sales_tf_Lag_52_Weeks'].fillna(X_all['Weekly_Sales_tf_y'], inplace=True)

    X_all = X_all[[col for col in X_all.columns if not col.endswith('_y')]]

    

## ffill remaining. Bad because departments and times aren't aligned (only 571 values --> < 0.5%) ##

X_all['Weekly_Sales_tf_Lag_52_Weeks'].fillna(method='ffill', inplace=True)

#############################



X_all.drop(['52_Week_Lag'], axis=1, inplace=True)



X_all[pd.to_datetime(X_all['Date']) >= datetime(2012, 11, 2)].isna().sum()
X_train_tf['Date2'] = pd.to_datetime(X_train_tf['Date'], utc = True)

X_test['Date2'] = pd.to_datetime(X_test['Date'], utc = True)



X_train_tf['Weekly_Sales_tf_Lag_52_Weeks'] = X_train_tf.merge(X_all, 

                                                   left_on=['Store', 'Dept', 'Date2'], 

                                                   right_on=['Store', 'Dept', 'Date2'],

                                                   how='inner')['Weekly_Sales_tf_Lag_52_Weeks']

X_test['Weekly_Sales_tf_Lag_52_Weeks'] = X_test.merge(X_all, 

                                                 left_on=['Store', 'Dept', 'Date2'], 

                                                 right_on=['Store', 'Dept', 'Date2'],

                                                 how='inner')['Weekly_Sales_tf_Lag_52_Weeks']



X_train_tf.drop(['Date2', 'outlier'], axis=1, inplace=True)

X_test.drop('Date2', axis=1, inplace=True)



X_test.isna().sum()
fig, axes = plt.subplots(ncols=5, figsize=(20,8))

sns.distplot(X_train['Weekly_Sales'], bins=10, ax=axes[0]).set_title('Weekly Sales')

sns.distplot(X_train['Weekly_Sales_Log'], bins=10, ax=axes[1]).set_title('Log(1+Weekly Sales)')

sns.distplot(X_train_tf['Weekly_Sales_Log'], bins=10, ax=axes[2]).set_title('Log(1+Weekly Sales)\nno outliers')

sns.distplot(X_train['Weekly_Sales_tf'], bins=10, ax=axes[3]).set_title('(1+Weekly Sales)\nQauntile Transformer')

sns.distplot(X_train_tf['Weekly_Sales_tf'], bins=10, ax=axes[4]).set_title('(1+Weekly Sales)\nQauntile Transformer\nno outliers')



plt.tight_layout()
## rename columns to remove `tf` ##



X_train_tf['Weekly_Sales'] = X_train['Weekly_Sales_tf']

X_train_tf['Weekly_Sales_Lag_52_Weeks'] = X_train_tf['Weekly_Sales_tf_Lag_52_Weeks']

X_train_tf.drop(['Weekly_Sales_tf', 'Weekly_Sales_tf_Lag_52_Weeks'], axis=1, inplace=True)



X_test['Weekly_Sales_Lag_52_Weeks'] = X_test['Weekly_Sales_tf_Lag_52_Weeks']

X_test.drop(['Weekly_Sales_tf_Lag_52_Weeks'], axis=1, inplace=True)



y_train_qt_tf = X_train_tf[['Store', 'Dept', 'Date', 'Weekly_Sales', 'Weekly_Sales_Lag_52_Weeks']]

X_train_tf.drop(['Weekly_Sales', 'Weekly_Sales_Log'], axis=1, inplace=True)

y_train_qt_tf.head()
print(X_train_tf.shape)



X_train_tf.head()
print(X_train_tf['Type'].value_counts(), '\n', X_test['Type'].value_counts())

print(X_train_tf['IsHoliday'].value_counts(), '\n', X_test['IsHoliday'].value_counts())
lbl_encoder = LabelEncoder()



X_train_tf['IsHoliday'] = X_train_tf['IsHoliday'].replace(True, 5).replace(False, 1).values # go off the custom weighted-mae function

X_train_tf['Type'] = lbl_encoder.fit_transform(X_train_tf['Type'])



X_test['IsHoliday'] = X_test['IsHoliday'].replace(True, 5).replace(False, 1).values # go off the custom weighted-mae function

X_test['Type'] = lbl_encoder.transform(X_test['Type'])
X_train_tf.head()
X_train_tf.isna().sum()
def weighted_mae_custom(y_true, y_pred):

    '''

    Custom weighting function as specified in the evaluation section.

    '''

    weights = X_train_tf['IsHoliday']

    sample_weights = pd.Series(weights.loc[y_true.index.values].values.reshape(-1)).dropna()

    return (1.0 / np.sum(sample_weights)) * np.sum(sample_weights * np.abs(y_true - y_pred))



weighted_mae = make_scorer(weighted_mae_custom)
X_train_tf.dtypes
params={'rf': {

#             'n_estimators': [100, 250, 500],

#             'max_depth': [1, 2, 3, 4],

#             'max_features': [2, 4, 6, 8]

            },

        'knn': {

#             'n_neighbors': [2, 3, 4],  

#             'p': [1,2],

            },

        'gb': {

#             'max_depth': [1, 2, 3, 4],

#             'learning_rate':[1e-3,1e-2,0.1,1]

            },

        'lr':{

            'fit_intercept': [True, False]

            },

        'lgbm':{

#             'learning_rate':[1e-3,1e-2,0.1,1],

#             'n_estimators': [100, 250, 500],

            },

        'xgb':{

            

            },

        }

        

models = {

          'lr': LinearRegression(n_jobs=-1),

          'knn': KNeighborsRegressor(n_jobs=-1),

          'rf': RandomForestRegressor(random_state=0, n_jobs=-1, n_estimators=100),

          'gb': GradientBoostingRegressor(random_state=0),

          'lgbm': LGBMRegressor(random_state=0),

          'xgb': XGBRegressor(nthread=-1, seed=0),

         }



best_params = {}

best_models = []



for name, model in models.items():

    cv1 = RepeatedKFold(n_splits=2, n_repeats=1,  random_state=0)                       

    gs_cv = GridSearchCV(model, 

                         params[name], 

                         scoring=weighted_mae,

                         cv=cv1,

                         n_jobs=-1,

                         iid=True,

                         verbose=2)

    

    gs_cv.fit(X_train_tf.drop('Date', axis=1).dropna(), y_train_qt_tf.dropna()['Weekly_Sales'])

    

    mean = abs(gs_cv.cv_results_['mean_test_score'][0])

    std = gs_cv.cv_results_['std_test_score'][0]

    

    best_params[name] = gs_cv.best_params_

    best_models.append(gs_cv.best_estimator_)

    

    print("Results for {}: {:.4f} ({:.4f}) [{:.4f}, {:.4f}] WMAE".format(name, 

                                                                         mean,

                                                                         std,

                                                                         mean - std,

                                                                         mean + std))
## run to get hyperparameters for `VotingRegressor` ##



best_params_2 = {}

for model,params in best_params.items():

    for param,value in best_params[model].items():

        best_params_2['{}__{}'.format(model, param)] = [value]

        

best_params_2
def voting_regressor(X, y):

    # br and lr similar performance, knn slightly worse, rf and gbr perform very badly. Use this to select weighting.

    v_reg = VotingRegressor(estimators=[('lr', LinearRegression(n_jobs=-1)),

                                        ('knn', KNeighborsRegressor(n_jobs=-1)),

                                        ('gb', GradientBoostingRegressor(random_state=0)),

#                                         ('rf', RandomForestRegressor(random_state=0, n_jobs=-1, n_estimators=100)),

                                        ('lgbm', LGBMRegressor(random_state=0)),

                                        ('xgb', XGBRegressor(nthread=-1, seed=0))

                                       ], 

                            n_jobs=-1,

                            weights=[0.10, 0.10, 0.20, 0.40, .20] ## specify weight given to each model in prediction (overweight rf & lgbm) ##

                           )

    

    cv1 = RepeatedKFold(n_splits=2, n_repeats=1,  random_state=0)                       

    gs_cv = GridSearchCV(v_reg, 

                         best_params_2, ## no parameters used in previous implementation, except logistic regression ##

                         scoring=weighted_mae,

                         cv=cv1, 

                         n_jobs=-1,

                         iid=True,

                         verbose=2)

    

    gs_cv.fit(X, y)



    mean = abs(gs_cv.cv_results_['mean_test_score'][0])

    std = gs_cv.cv_results_['std_test_score'][0]

    

    print("Results for {}: {:.4f} ({:.4f}) [{:.4f}, {:.4f}] accuracy".format('VotingRegressor', 

                                                                             mean,

                                                                             std,

                                                                             mean - std,

                                                                             mean + std))

    

    return gs_cv.best_estimator_
vot_reg = voting_regressor(X_train_tf.drop('Date', axis=1).dropna(), y_train_qt_tf.dropna()['Weekly_Sales'])
## takes too long ##
def weighted_mae_keras(weights):

    def loss(y_true, y_pred):

        return (1.0 / np.sum(weights)) * keras.backend.sum(weights * keras.backend.abs(y_true - y_pred))

    

    return loss
# def weighted_mae_keras(y_true, y_pred):

#     return keras.losses.mean_absolute_error(y_true, y_pred)



def build_model(X, weights):

    model = keras.Sequential([

        layers.Dense(2048, activation=tf.nn.leaky_relu, input_shape=[X.shape[1]]),

        layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.Dense(1024, activation=tf.nn.leaky_relu),

        layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.Dense(512, activation=tf.nn.leaky_relu),

        layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.Dense(128, activation=tf.nn.leaky_relu),

        layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.Dense(1)

    ])



    model.compile(

                  loss=weighted_mae_keras(weights),

#                   loss='mae',

                  optimizer='adam',)

#                   metrics=[weighted_mae_keras(weights)])

    return model
weights = X_train_tf['IsHoliday'].values



model = build_model(X_train_tf.drop('Date', axis=1).dropna(), weights) ## drop rows with NaN lagged 1-year `Weekly_Sales` ##
mm_scaler = MinMaxScaler()

X_train_tf_scaled = pd.DataFrame(mm_scaler.fit_transform(X_train_tf.drop('Date', axis=1).dropna()), columns=X_train_tf.drop('Date', axis=1).columns)



EPOCHS = 1



history = model.fit(X_train_tf_scaled[:1000], 

                    y_train_qt_tf.dropna()['Weekly_Sales'][:1000],

                    epochs=EPOCHS, 

#                     callbacks=[es],

                    validation_split=0.25,

                    verbose=1,

#                     verbose=2,

                    workers=10,

                    use_multiprocessing=True

                   )
# hist = pd.DataFrame(history.history)

# hist.plot()
## create 1-year lag value of store sales (can't use 1-week as test set doesn't have any `Weekly_Sales` ##

## X_train ends at 2012-10-26 and X_test ends at 2013-07-26, so no NaN values for `Weekly_Sales_Lag` in X_test ##
# X_all = pd.concat([X_train, X_test])

# X_all.tail()
best_models[2]
X_test.drop('Date', axis=1).isna().sum()
## QuantileTransformer ##

X_test['Weekly_Sales'] = qt.inverse_transform(best_models[2].predict(X_test.drop('Date', axis=1)).reshape(-1, 1)) + 1

X_test.head()
'''

Id,Weekly_Sales

1_1_2012-11-02,0

'''

df_pred = pd.DataFrame(columns=['Id', 'Weekly_Sales'])

df_pred['Id'] = X_test['Store'].astype(str) + '_' + X_test['Dept'].astype(str) + '_' + X_test['Date'].astype(str)

df_pred['Weekly_Sales'] = X_test['Weekly_Sales']



df_pred.head()
df_pred.to_csv('submission.csv', index=False)