# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import RepeatedKFold, cross_val_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import VotingRegressor

from sklearn.model_selection import cross_validate, RepeatedKFold



from xgboost import XGBRegressor



sns.set_style('darkgrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

np.random.seed(0)



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

X_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



df_train = df_train.sample(frac=1, random_state=0).reset_index(drop=True)

X_test = X_test.sample(frac=1, random_state=0).reset_index(drop=True)



X_train = df_train.drop('SalePrice', axis=1)

y_train = pd.DataFrame(df_train['SalePrice'], columns=['SalePrice'])



X_train.shape, X_test.shape
# !pip install pandas-profiling

# import pandas_profiling



# df_train.profile_report(style={'full_width':True})
X_train.isna().sum().sort_values()[-20:]
X_test.isna().sum().sort_values()[-20:]
train_cols_cat_na_sum = X_train.isna().sum().sort_values()

test_cols_cat_na_sum = X_train.isna().sum().sort_values()



train_cols_cat_na_sum = list(train_cols_cat_na_sum[train_cols_cat_na_sum > 250].keys())

test_cols_cat_na_sum = list(test_cols_cat_na_sum[test_cols_cat_na_sum > 250].keys())



cols_cat_na = list(set(train_cols_cat_na_sum + test_cols_cat_na_sum))



print('Cols with number of missing values > 250: ', cols_cat_na)



X_train.drop(cols_cat_na, axis=1, inplace=True)

X_test.drop(cols_cat_na, axis=1, inplace=True)
## Remove cols from X_train and X_test where categorical values not contained in train and test set ##

cols_obj = [col for col in X_train.columns if df_train[col].dtype == object]

bad_lbl_cols = [col for col in cols_obj if list(set(X_train[col]) - set(X_test[col]))]

print('Categorical columns with values not in both train and test set: ', bad_lbl_cols)



X_train.drop(bad_lbl_cols, axis=1, inplace=True)

X_test.drop(bad_lbl_cols, axis=1, inplace=True)
# drop unneccessary column Id #

X_train.drop('Id', axis=1, inplace=True)

test_ids = X_test['Id'].values

X_test.drop('Id', axis=1, inplace=True)
cols_num = [col for col in X_train.columns if X_train[col].dtype in [float, int]]

ncols = len(cols_num) // 4

fig, axes = plt.subplots(ncols=ncols, nrows=5, figsize=(30,16))



i = 1

for j, col in enumerate(cols_num):

    sns.distplot(df_train[col], bins=10, ax=axes[i-1][j % ncols])



    if j % ncols == (ncols - 1):

        i += 1

        

plt.tight_layout()
sns.distplot(df_train['SalePrice'], bins=10)
## Create new features ##

sf_cols = [x for x in X_train.columns if 'sf' in x.lower()]

print('Square foot columns: ', sf_cols)



# only considers internal home square footages. Only need `TotalBsmtSF` #

sf_cols_2 = [x for x in sf_cols if x not in ['LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Total_SF']]

X_train['Total_SF'] = X_train[sf_cols_2].sum(axis=1)

X_test['Total_SF'] = X_test[sf_cols_2].sum(axis=1)



# calc # bedrooms per room #

X_train['bedrooms_per_room'] = X_train['BedroomAbvGr'] / X_train['TotRmsAbvGrd']

X_test['bedrooms_per_room'] = X_test['BedroomAbvGr'] / X_test['TotRmsAbvGrd']





cols_to_binary = ['PoolArea', '2ndFlrSF', 'GarageArea', 'TotalBsmtSF', 'Fireplaces']

X_train['haspool'] = X_train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

X_train['has2ndfloor'] = X_train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

X_train['hasgarage'] = X_train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

X_train['hasbsmt'] = X_train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

X_train['hasfireplace'] = X_train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



X_test['haspool'] = X_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

X_test['has2ndfloor'] = X_test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

X_test['hasgarage'] = X_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

X_test['hasbsmt'] = X_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

X_test['hasfireplace'] = X_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



X_train.drop(cols_to_binary, axis=1, inplace=True)

X_test.drop(cols_to_binary, axis=1, inplace=True)
cols_num = [col for col in X_train.columns if X_train[col].dtype in [float, int]]

cols_obj = [col for col in X_train.columns if X_train[col].dtype == object]



col_obj_cnts = {col: X_train[col].nunique() for col in cols_obj}
cols_num_cnts = X_train[cols_num].nunique()



# 140 is a good cutoff from years and identifiers, to numerical measurements like square foot

num_cols_mean = list(cols_num_cnts[cols_num_cnts > 140].keys())

num_cols_med  = list(cols_num_cnts[cols_num_cnts <= 140].keys())



print('Numeric columns mean fill:\n', num_cols_mean)

print('\nNumeric columns median fill:\n', num_cols_med)
OH_cols = [k for k,v in col_obj_cnts.items() if v < 6] # only OHE variables w/ less than 6 unique values per column

LE_cols = list(set(cols_obj) - set(OH_cols))



print('OH cols:\n', OH_cols)

print('\nLE cols:\n', LE_cols)
def transform_data(X):

    # two steps: 

    # (1) Fill in missing values using SimpleImputer

    # (2) One Hot Encode the variables, creating new columns for each unique type

    OH_transformer = Pipeline(steps=[('imputer1', SimpleImputer(strategy='most_frequent')),

                                     ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse=False)) # ignore errors when calling 'transform', sparse=False returns np.array

    ])



    # Same as OH_transformer except using LabelEncoder instead --> high-cardinality columns

    LE_transformer = Pipeline(steps=[('imputer2', SimpleImputer(strategy='most_frequent')),

                                     ('lbl_enc', OrdinalEncoder())

    ])



    # Fill in missing values with mean

    num_mean_transformer = SimpleImputer(strategy='mean')

    num_med_transformer  = SimpleImputer(strategy='median')



    # Bundle preprocessing for numerical and two categorical groups data

    preprocessor = ColumnTransformer(

        transformers=[

            ('oh', OH_transformer, OH_cols),

            ('le', LE_transformer, LE_cols),

            ('num_mean', num_mean_transformer, num_cols_mean),

            ('num_med', num_med_transformer, num_cols_med),

        ])



    pipe = Pipeline(steps=[ ('preprocessor', preprocessor),

    #                         ('scaler', StandardScaler())

    ])

    

    X_copy = X.copy(deep=True)

    oh_col_names = OH_transformer.fit(X_copy[OH_cols])['one_hot'].get_feature_names(OH_cols)

    cols = list(oh_col_names) + list([x for x in X_copy.columns if x not in OH_cols])



    X_copy = pipe.fit_transform(X_copy)

    X_copy = pd.DataFrame(X_copy, columns=cols)

    

    return X_copy
X_train_tf = transform_data(X_train)

X_test_tf = transform_data(X_test)



print(X_train_tf.shape, X_test_tf.shape)

X_train_tf.head()
from scipy.stats import normaltest

## H0: normally distributed ##

## H1: not normally distributed ##



k2, p = normaltest(X_train_tf)

sum([1 for x in p if x < 0.05]), X_train_tf.shape[1]
from sklearn.preprocessing import power_transform

## https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html ##

## Power transforms applied to make data more Gaussian-like ##



## add 1 b/c all values must be positive for box-cox transformation ##

## standard scale to mean 0 unit variance 1 ##

X_train_tf = pd.DataFrame(power_transform(1 + X_train_tf, method='box-cox', standardize=True, copy=True), columns=X_train_tf.columns)

X_test_tf = pd.DataFrame(power_transform(1 + X_test_tf, method='box-cox', standardize=True, copy=True), columns=X_test_tf.columns)
k2, p = normaltest(X_train_tf)

sum([1 for x in p if x < 0.05]), X_train_tf.shape[1]
X_train_tf = transform_data(X_train)

X_test_tf = transform_data(X_test)



robust_scaler = RobustScaler()



robust_scaler.fit(X_train_tf)

X_train_tf = pd.DataFrame(robust_scaler.transform(X_train_tf), columns=X_train_tf.columns)

robust_scaler.fit(X_test_tf)

X_test_tf = pd.DataFrame(robust_scaler.transform(X_test_tf), columns=X_test_tf.columns)
k2, p = normaltest(X_train_tf)

sum([1 for x in p if x < 0.05]), X_train_tf.shape[1]
X_train_tf = transform_data(X_train)

X_test_tf = transform_data(X_test)



cols_to_log = [col for col in X_train_tf.columns if ((X_train_tf[col].dtype in [float, int]) and (X_train_tf[col].nunique() > 140))]

for col in cols_to_log:

    X_train_tf[col] = np.log1p(X_train_tf[col]).values.ravel()

    X_test_tf[col] = np.log1p(X_test_tf[col]).values.ravel()    

    

len(cols_to_log)
k2, p = normaltest(X_train_tf)

sum([1 for x in p if x < 0.05]), X_train_tf.shape[1]
X_train_tf = pd.DataFrame(power_transform(1 + X_train_tf, method='box-cox', standardize=True, copy=True), columns=X_train_tf.columns)

X_test_tf = pd.DataFrame(power_transform(1 + X_test_tf, method='box-cox', standardize=True, copy=True), columns=X_test_tf.columns)



k2, p = normaltest(X_train_tf)

sum([1 for x in p if x < 0.05]), X_train_tf.shape[1]
fig, axes = plt.subplots(ncols=2, figsize=(12,5))



y_train_tf = np.log1p(y_train).values.ravel()

sns.distplot(y_train, bins=10, ax=axes[0]).set_title('Normal SalePrice')

sns.distplot(y_train_tf, bins=10, ax=axes[1]).set_title('(Log+1) SalePrice')
def lasso_feat_select(X, y):

    params={'alpha': [1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100]}

    lasso = Lasso(random_state=0)



    gs_cv = GridSearchCV(lasso, params, cv=5, scoring='neg_mean_squared_error')

    gs_cv.fit(X, y)

    lasso_best = gs_cv.best_estimator_



    coef = pd.Series(lasso_best.coef_, index = X.columns)

    coef = coef[abs(coef) > 1e-3]



    print('Coefficients went from {} to {} after Lasso Regression.'.format(X.shape[1], len(coef)))

    X_new = X[coef.index]

    

    return X_new, coef

    

X_train_tf, coef = lasso_feat_select(X_train_tf, y_train_tf)

X_test_tf = X_test_tf[coef.index]



imp_coef = coef.sort_values()

imp_coef.plot(kind = "barh", figsize=(10,10))

plt.title("Feature importance using Lasso Model")
## custom RMSE loss function

def root_mean_squared_error_custom(y_true, y_predicted):

    error = np.sqrt(np.mean((y_true - y_predicted)**2))

    return error



from sklearn.metrics import make_scorer

root_mean_squared_error = make_scorer(root_mean_squared_error_custom, greater_is_better=False)
from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import BayesianRidge, SGDRegressor



params={'rf': {

            'n_estimators': [100, 250, 500],

            'max_depth': [1, 2, 3, 4],

            'max_features': [2, 4, 6, 8]

            },

        'knn': {

            'n_neighbors': [2, 3, 4],  

            'p': [1,2],

            },

        'gb': {

            'max_depth': [1, 2, 3, 4],

            'learning_rate':[1e-3,1e-2,0.1,1]

            },

        'lr':{

            'fit_intercept': [True, False]

            },

        'br':{

            'n_iter': [100, 250, 500],

            'alpha_1': [1e-10, 1e-8, 1e-6],

            'alpha_2': [1e-4, 1e-2, 1],

            'lambda_1': [1e-10, 1e-8, 1e-6],

            'lambda_2': [1e-4, 1e-2, 1],

            'fit_intercept': [True, False]

            },

        'sgd':{

            'penalty': ['l2', 'l1', 'elasticnet'],

            'alpha': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1],

            'l1_ratio': [0.01, 0.2, 0.4, 0.6, 0.8, 1.0],

            'fit_intercept': [True, False]

            },

        }

        

models = {

          'lr': LinearRegression(n_jobs=-1),

          'knn': KNeighborsRegressor(n_jobs=-1),

          'rf': RandomForestRegressor(random_state=0, n_jobs=-1),

          'gb': GradientBoostingRegressor(random_state=0),

          'br': BayesianRidge(),

#           'sgd': SGDRegressor(random_state=0),

         }



best_params = {}

best_models = []

for name, model in models.items():

    cv1 = RepeatedKFold(n_splits=3, n_repeats=6,  random_state=0)                       

    gs_cv = GridSearchCV(model, 

                         params[name], 

                         scoring=root_mean_squared_error,

                         cv=cv1,

                         n_jobs=-1,

                         iid=True)

    

    gs_cv.fit(X_train_tf, y_train_tf)

    

    mean = abs(gs_cv.cv_results_['mean_test_score'][0])

    std = gs_cv.cv_results_['std_test_score'][0]

    

    best_params[name] = gs_cv.best_params_

    best_models.append(gs_cv.best_estimator_)

    

    print("Results for {}: {:.4f} ({:.4f}) [{:.4f}, {:.4f}] MSE".format(name, 

                                                                         mean,

                                                                         std,

                                                                         mean - std,

                                                                         mean + std))
best_params_2 = {}

for model,params in best_params.items():

    for param,value in best_params[model].items():

        best_params_2['{}__{}'.format(model, param)] = [value]

        

best_params_2
def voting_regressor(X, y):

    # br and lr similar performance, knn slightly worse, rf and gbr perform very badly. Use this to select weighting.

    v_reg = VotingRegressor(estimators=[('br', BayesianRidge()),

                                        ('lr', LinearRegression(n_jobs=-1)),

                                        ('knn', KNeighborsRegressor(n_jobs=-1)),

                                        ('rf', RandomForestRegressor(random_state=0, n_jobs=-1)),

                                        ('gb', GradientBoostingRegressor(random_state=0)),

#                                         ('sgd', SGDRegressor(random_state=0)),

                                       ], 

                            n_jobs=-1,

                            weights=[0.30, 0.30, 0.20, 0.10, 0.10])

    

    cv1 = RepeatedKFold(n_splits=3, n_repeats=6,  random_state=0)                       

    gs_cv = GridSearchCV(v_reg, 

                         best_params_2,

                         scoring=root_mean_squared_error,

                         cv=cv1, 

                         n_jobs=-1,

                         iid=True)

    

    gs_cv.fit(X, y)



    mean = abs(gs_cv.cv_results_['mean_test_score'][0])

    std = gs_cv.cv_results_['std_test_score'][0]

    

    print("Results for {}: {:.4f} ({:.4f}) [{:.4f}, {:.4f}] accuracy".format('VotingRegressor', 

                                                                             mean,

                                                                             std,

                                                                             mean - std,

                                                                             mean + std))

    

    return gs_cv.best_estimator_
vot_reg = voting_regressor(X_train_tf, y_train_tf)
xgb = XGBRegressor(nthread=-1, seed=0)



cv1 = RepeatedKFold(n_splits=3, n_repeats=6,  random_state=0)                       

gs_cv = GridSearchCV(xgb, 

                     {},

                     scoring=root_mean_squared_error,

                     cv=cv1, 

                     n_jobs=-1,

                     iid=True)



gs_cv.fit(X_train_tf, y_train_tf)



xgb = gs_cv.best_estimator_



mean = abs(gs_cv.cv_results_['mean_test_score'][0])

std = gs_cv.cv_results_['std_test_score'][0]



print("Results for {}: {:.4f} ({:.4f}) [{:.4f}, {:.4f}] accuracy".format('XGBRegressor', 

                                                                         mean,

                                                                         std,

                                                                         mean - std,

                                                                         mean + std))
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
def process_features(X, y=None, features_idx=[]):

    ## Do not standardize, instead use MinMaxScaler (all values in range [0,1]) for NN ##

    for col in cols_to_log:

        X[col] = np.log1p(X[col]).values.ravel()

        

    X_new = pd.DataFrame(power_transform(1 + X, method='box-cox', standardize=False, copy=True), columns=X.columns)

    

    mm_scaler = MinMaxScaler()

    X_new = pd.DataFrame(mm_scaler.fit_transform(X_new), columns=X.columns)

    

    return X_new



## Fill in NaN values, LabelEncode, OneHotEncode ##

X_train_tf_2 = transform_data(X_train)

X_test_tf_2 = transform_data(X_test)



## Scale Data ##

X_train_tf_2 = process_features(X_train_tf_2, y_train_tf)

X_test_tf_2 = process_features(X_test_tf_2, y_train_tf)



## Feature Selection ##

X_train_tf_2, coef = lasso_feat_select(X_train_tf_2, y_train_tf)

X_test_tf_2 = X_test_tf_2[coef.index]
def root_mean_squared_error(y_true, y_pred):

        return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true))) 



def build_model(X):

    model = keras.Sequential([

        layers.Dense(128, activation=tf.nn.leaky_relu, kernel_initializer=keras.initializers.TruncatedNormal, input_shape=[X.shape[1]]),

        layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.Dense(64, activation=tf.nn.leaky_relu, kernel_initializer=keras.initializers.TruncatedNormal),

        layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.Dense(32, activation=tf.nn.leaky_relu, kernel_initializer=keras.initializers.TruncatedNormal),

        layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.Dense(1)

    ])



    model.compile(loss=root_mean_squared_error,

                  optimizer='adam',

                  metrics=[root_mean_squared_error])

    return model
model = build_model(X_train_tf_2)
EPOCHS = 1000



'''

To avoid overfitting the training set, interrupt training when its performance on the validation set starts dropping.

Patience - number of epochs that produced the monitored quantity with no improvement after which training will be stopped.

Reference: https://keras.io/callbacks/

'''

es = callbacks.EarlyStopping(monitor='root_mean_squared_error', min_delta=1e-4, patience=50,

                             verbose=1, mode='min', baseline=None, restore_best_weights=True)



''' 

Learning Rate -  set too high, training diverges 

              -  set too low, training eventually converges to the optimum, but takes long time

              -  solution: start with high LR, then reduce

Reference: https://keras.io/callbacks/

'''

rlr = callbacks.ReduceLROnPlateau(monitor='root_mean_squared_error', factor=0.5,

                                  patience=50, min_delta=1e-4, min_lr=1e-4, mode='min', verbose=0)



rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=0)

models = []



for index, _ in rkf.split(X_train_tf_2, y_train_tf):

    X_train_tf_2_subset = X_train_tf_2.loc[index]

    y_train_tf_subset = y_train_tf[index]

    print(X_train_tf_2_subset.shape, y_train_tf_subset.shape)

#     print(index)



    model = build_model(X_train_tf_2_subset)

    history = model.fit(X_train_tf_2_subset, 

                        y_train_tf_subset,

                        epochs=EPOCHS, 

                        callbacks=[es, rlr],

    #                     callbacks=[es],

                        validation_split=0.25,

                        verbose=0,

    #                     verbose=2,

                       )

    models.append([model, history])
plt.figure(figsize=(10,6))

for i, model_params in enumerate(models):

    hist = pd.DataFrame(model_params[1].history)

    hist['val_root_mean_squared_error'][-50:].plot(title='NN RMSE', label=i+1)

    plt.legend()
#Make predictions using the features from the test data set



predictions = []

# NN models

for model_params in models:

    predictions.append([x[0] for x in np.expm1(model_params[0].predict(X_test_tf_2))])



# Voting regressor #

predictions.append(np.expm1(vot_reg.predict(X_test_tf)))

# XGBRegressor #

predictions.append(np.expm1(xgb.predict(X_test_tf)))



predictions = np.array(predictions)

predictions = list(np.average(predictions, axis=0))

    

predictions[-5:]
df_final = pd.DataFrame(np.column_stack((test_ids, predictions)), columns=['Id', 'SalePrice'])

df_final = df_final.astype({'Id': 'Int32', 'SalePrice': 'float'})

df_final = df_final.sort_values(by='Id').reset_index(drop=True)

df_final.head()
df_final.to_csv('submission.csv', index=False)