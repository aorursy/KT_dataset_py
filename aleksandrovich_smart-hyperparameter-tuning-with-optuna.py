import sys

import warnings



import lightgbm as lgb

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import random

import seaborn as sns



from mlxtend.regressor import StackingCVRegressor

from scipy.stats import skew

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import ElasticNetCV

from sklearn.linear_model import LassoCV

from sklearn.linear_model import RidgeCV

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVR

from xgboost import XGBRegressor



from keras.layers import Activation

from keras.layers import BatchNormalization

from keras.layers import Dense

from keras.layers import Dropout

from keras.models import Sequential

from keras.optimizers import Adam





import optuna



print("Imports have been set")

random.seed(42)



# Disabling warnings

if not sys.warnoptions:

    warnings.simplefilter("ignore")
# Reading the training/val data and the test data

X = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')



# Rows before:

rows_before = X.shape[0]

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

rows_after = X.shape[0]

print("Rows containing NaN in SalePrice were dropped: " + str(rows_before - rows_after))



# Let's look at the target variable distribution

# sns.distplot(a=X['SalePrice'], label="Target variable distribution", kde=False)

# sns.barplot(x=X.index, y=X['SalePrice'])

# plt.show()

# print("Well, looks like it's shuffled properly")
# Logarithming target variable in order to make distribution better

X['SalePrice'] = np.log1p(X['SalePrice'])

y = X['SalePrice'].reset_index(drop=True)

train_features = X.drop(['SalePrice'], axis=1)



# Let's see what happens after

sns.distplot(a=y, label="Target variable distribution", kde=False)

plt.show()
# concatenate the train and the test set as features for tranformation to avoid mismatch

features = pd.concat([train_features, X_test]).reset_index(drop=True)

print('Features size:', features.shape)
nan_count_table = (features.isnull().sum())

nan_count_table = nan_count_table[nan_count_table > 0].sort_values(ascending=False)

print("\nColums containig NaN: ")

print(nan_count_table)



columns_containig_nan = nan_count_table.index.to_list()

print("\nWhat values they contain: ")

print(features[columns_containig_nan])
for column in columns_containig_nan:



    # populating with 0

    if column in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

                  'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'TotalBsmtSF',

                  'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold',

                  'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea']:

        features[column] = features[column].fillna(0)



    # populate with 'None'

    if column in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', "PoolQC", 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

                  'BsmtFinType2', 'Neighborhood', 'BldgType', 'HouseStyle', 'MasVnrType', 'FireplaceQu', 'Fence', 'MiscFeature']:

        features[column] = features[column].fillna('None')



    # populate with most frequent value for cateforic

    if column in ['Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'RoofStyle',

                  'Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'RoofMatl', 'ExterQual', 'ExterCond',

                  'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'PavedDrive', 'SaleType', 'SaleCondition']:

        features[column] = features[column].fillna(features[column].mode()[0])



# MSSubClass: Numeric feature. Identifies the type of dwelling involved in the sale.

#     20  1-STORY 1946 & NEWER ALL STYLES

#     30  1-STORY 1945 & OLDER

#     40  1-STORY W/FINISHED ATTIC ALL AGES

#     45  1-1/2 STORY - UNFINISHED ALL AGES

#     50  1-1/2 STORY FINISHED ALL AGES

#     60  2-STORY 1946 & NEWER

#     70  2-STORY 1945 & OLDER

#     75  2-1/2 STORY ALL AGES

#     80  SPLIT OR MULTI-LEVEL

#     85  SPLIT FOYER

#     90  DUPLEX - ALL STYLES AND AGES

#    120  1-STORY PUD (Planned Unit Development) - 1946 & NEWER

#    150  1-1/2 STORY PUD - ALL AGES

#    160  2-STORY PUD - 1946 & NEWER

#    180  PUD - MULTILEVEL - INCL SPLIT LEV/FOYER

#    190  2 FAMILY CONVERSION - ALL STYLES AND AGES



# Stored as number so converted to string.

features['MSSubClass'] = features['MSSubClass'].apply(str)

features["MSSubClass"] = features["MSSubClass"].fillna("Unknown")

# MSZoning: Identifies the general zoning classification of the sale.

#    A    Agriculture

#    C    Commercial

#    FV   Floating Village Residential

#    I    Industrial

#    RH   Residential High Density

#    RL   Residential Low Density

#    RP   Residential Low Density Park

#    RM   Residential Medium Density



# 'RL' is by far the most common value. So we can fill in missing values with 'RL'

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# LotFrontage: Linear feet of street connected to property

# Groupped by neighborhood and filled in missing value by the median LotFrontage of all the neighborhood

# TODO may be 0 would perform better than median?

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# LotArea: Lot size in square feet.

# Stored as string so converted to int.

features['LotArea'] = features['LotArea'].astype(np.int64)

# Alley: Type of alley access to property

#    Grvl Gravel

#    Pave Paved

#    NA   No alley access



# So. If 'Street' made of 'Pave', so it would be reasonable to assume that 'Alley' might be 'Pave' as well.

features['Alley'] = features['Alley'].fillna('Pave')

# MasVnrArea: Masonry veneer area in square feet

# Stored as string so converted to int.

features['MasVnrArea'] = features['MasVnrArea'].astype(np.int64)
features['YrBltAndRemod'] = features['YearBuilt'] + features['YearRemodAdd']

features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



# If area is not 0 so creating new feature looks reasonable

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



print('Features size:', features.shape)
nan_count_train_table = (features.isnull().sum())

nan_count_train_table = nan_count_train_table[nan_count_train_table > 0].sort_values(ascending=False)

print("\nAre no NaN here now: " + str(nan_count_train_table.size == 0))
numeric_columns = [cname for cname in features.columns if features[cname].dtype in ['int64', 'float64']]

print("\nColumns which are numeric: " + str(len(numeric_columns)) + " out of " + str(features.shape[1]))

print(numeric_columns)



categoric_columns = [cname for cname in features.columns if features[cname].dtype == "object"]

print("\nColumns whice are categoric: " + str(len(categoric_columns)) + " out of " + str(features.shape[1]))

print(categoric_columns)



skewness = features[numeric_columns].apply(lambda x: skew(x))

print(skewness.sort_values(ascending=False))



skewness = skewness[abs(skewness) > 0.5]

features[skewness.index] = np.log1p(features[skewness.index])

print("\nSkewed values: " + str(skewness.index))
# Kind of One-Hot encoding

final_features = pd.get_dummies(features).reset_index(drop=True)



# Spliting the data back to train(X,y) and test(X_sub)

X = final_features.iloc[:len(y), :]

X_test = final_features.iloc[len(X):, :]



# Spltting X and y to train and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)



print("Shape of X_train: " + str(X_train.shape) + ", shape of y_train: " + str(y_train.shape))

print("Shape of X_valid: " + str(X_valid.shape) + ", shape of y_valid: " + str(y_valid.shape))
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



# check maybe 10 kfolds would be better

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)



# Kernel Ridge Regression : made robust to outliers

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))



# LASSO Regression : made robust to outliers

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=14, cv=kfolds))



# Elastic Net Regression : made robust to outliers

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))



# Gradient Boosting for regression

gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10,

                                   loss='huber', random_state=5)



# LightGBM regressor

dtrain = lgb.Dataset(X_train, label=y_train)

lgbm_params = {

    'objective': 'regression',

    'metric': 'mean_absolute_error',

    'lambda_l1': 0.009917563046305308,

    'lambda_l2': 0.0005854111105267089,

    'num_leaves': 179,

    'learning_rate': 0.08994549293068982,

    'n_estimators': 1780,

    'feature_fraction': 0.6669586810450638,

    'bagging_fraction': 0.6225238656510562,

    'bagging_freq': 4,

    'min_child_samples': 5}





# optimal parameters, received from CV

c_grid = {"n_estimators": [1000],

          "early_stopping_rounds": [1],

          "learning_rate": [0.1]}

xgb_regressor = XGBRegressor(objective='reg:squarederror', eval_metric='mae')

xgb_r = GridSearchCV(estimator=xgb_regressor,

                     param_grid=c_grid,

                     cv=kfolds)



# stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, lgbm, gboost),

#                                 meta_regressor=elasticnet,

#                                 use_features_in_secondary=True)



svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
def mae(y_actual, y_pred):

    return mean_absolute_error(np.expm1(y_actual), np.expm1(y_pred))
# sampler = TPESampler(seed=10) # for reproducibility

# def objective(trial):

#     dtrain = lgb.Dataset(X_train, label=y_train)

    

#     param = {

#         'objective': 'regression',

#         'metric': 'mean_absolute_error',

#         'verbosity': -1,

#         'boosting_type': 'gbdt',

#         'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),

#         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),

#         'num_leaves': trial.suggest_int('num_leaves', 2, 512),

#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),

#         'n_estimators': trial.suggest_int('n_estimators', 700, 3000),

#         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),

#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),

#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

#     }



#     gbm = lgb.train(param, dtrain)

#     return mean_absolute_error(np.expm1(y_valid), np.expm1(gbm.predict(X_valid)))



# study = optuna.create_study(direction='minimize', sampler=sampler)

# study.optimize(objective, n_trials=100)
# THIS VERSION SUCKS

import lightgbm as lgbm

from optuna.samplers import TPESampler



sampler = TPESampler(seed=10) # for reproducibility

def objective(trial):

    

    # dtrain = lgb.Dataset(X_train, label=y_train)

    

    param = {

        'objective': 'regression',

        'metric': 'mean_absolute_error',

        'verbosity': -1,

        'boosting_type': 'gbdt',

        # 'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 2000),

        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),

        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),

        'num_leaves': trial.suggest_int('num_leaves', 2, 512),

        # 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),

        'learning_rate': 0.01,

        'n_estimators': trial.suggest_int('n_estimators', 700, 3000),

        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),

        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),

        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

    }

    lgbm_regr = lgbm.LGBMRegressor(**param)

    gbm_2 = lgbm_regr.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    return mean_absolute_error(np.expm1(y_valid), np.expm1(gbm_2.predict(X_valid)))



study = optuna.create_study(direction='minimize', sampler=sampler)

study.optimize(objective, n_trials=100)
is_NN_on = False



if is_NN_on:

    model_nn = Sequential()



    model_nn.add(Dense(1028, input_dim=X_train.shape[1], init='he_normal'))

    model_nn.add(BatchNormalization())

    model_nn.add(Activation('relu'))

    model_nn.add(Dropout(0.2))



    model_nn.add(Dense(128, init='he_normal'))

    model_nn.add(BatchNormalization())

    model_nn.add(Activation('relu'))

    model_nn.add(Dropout(0.2))



    model_nn.add(Dense(1, init='he_normal'))

    #model_nn.add(BatchNormalization())

    #model_nn.add(Activation('sigmoid'))



    opt = Adam(lr=0.00001)

    model_nn.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae'])



    print('Neural Network is fitting now...')



    nn_history = model_nn.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=10, epochs=600)



    val_acc = nn_history.history['val_mae']

    acc = nn_history.history['mae']



    epochs = range(1, len(val_acc) + 1)



    plt.figure(figsize=(15,10))

    plt.ylim(0, 1.0)

    plt.plot(epochs, val_acc, label='Validation mae')

    plt.plot(epochs, acc, label='Train mae')

    plt.xlabel('Epochs')

    plt.ylabel('MAE')

    plt.legend()



    plt.show()
# print('lgbm is fitting now...')

# lgbm = lgb.train(lgbm_params, dtrain)
print('lgbm is fitting now...')

import lightgbm as lgbm

param = {'lambda_l1': 0.001040845213184379, 'lambda_l2': 1.0320042952640715e-08, 'num_leaves': 450, 'n_estimators': 992, 'feature_fraction': 0.5307283632444058, 'bagging_fraction': 0.48228480181276895, 'bagging_freq': 3, 'min_child_samples': 5}

lgbm_regr = lgbm.LGBMRegressor(**param)

lgbm = lgbm_regr.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric= 'mean_absolute_error', verbose=False)
MODELS = {

    # 'stack_gen': (stack_gen, 0.20),

    'svr': (svr, 0.21),

    'elastic': (elasticnet, 0.20),

    'gboost': (gboost, 0.20),

    'lgbm': (lgbm, 0.20),

    'lasso': (lasso, 0.19),

    #'ridge': (ridge, 0.16638505996865854),

    # 'xgb_r': (xgb_r, 0.10)

}
excluding_list = [lgbm]



print('Fitting our models ensemble: ')

for modelname, model in MODELS.items():

    if model[0] not in excluding_list:

        print(str(modelname), 'is fitting now...')

        model[0].fit(X_train, y_train)
print('Models evaluating: ')

scores = {}

for modelname, model in MODELS.items():

    score = mae(y_valid, model[0].predict(X_valid))

    print(modelname, "score: {:.4f}".format(score))

    scores[modelname] = 1/score
print('Optimal coefficients based on score: \n')

scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}



vals = np.fromiter(scores.values(), dtype=float)



for key in scores:

    print(key, ': ', scores[key] / sum(vals))
#  Last successful: 0.11663

def blend_models(models, x):

    output = np.zeros(x.shape[0])

    for blend_modelname, blend_model in models.items():

        output = np.add(output, blend_model[1] * blend_model[0].predict(x))

    return output



print('MAE score on validation data:')

print('Ensemble: ', mae(y_valid, blend_models(MODELS, X_valid)))



if is_NN_on:

    print('Neural network: ', mae(y_valid, pd.Series(model_nn.predict(X_valid).reshape(1, X_valid.shape[0])[0]).values))
submission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')

submission.iloc[:, 1] = np.expm1(blend_models(MODELS, X_test))

submission.to_csv("submission.csv", index=False)



print("Submission file is formed")