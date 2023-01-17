import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy



from IPython.display import display



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error



from sklearn.linear_model import ElasticNetCV

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



import warnings

warnings.filterwarnings('ignore')
!ls "../input/"
# Reading train&test data



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print("Train set size:", train.shape)

print("Test set size:", test.shape)
# Drop the id's from both dataset.



train_ID = train['Id']

test_ID = test['Id']



train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
# Evaluation metric is root mean squared log error. So we are taking the log of dependent variable.



train.SalePrice = np.log(train.SalePrice)

y = train.SalePrice.reset_index(drop=True)

train_features = train.drop(['SalePrice'], axis=1)

test_features = test
# Concatenate train and test features in order to preprocess both of them equally.



features = pd.concat([train_features, test_features]).reset_index(drop=True)

print(features.shape)
# Print numeric and categoric columns.



numeric_columns = features.select_dtypes(exclude=['object']).columns

categoric_columns = features.select_dtypes(include=['object']).columns

print(numeric_columns)

print(categoric_columns)



assert len(numeric_columns) + len(categoric_columns) == features.shape[1], "Some columns are missing?"
def nulls(df):

    null = (features.isnull().sum() / len(features)).sort_values(ascending=False)

    null = null[null > 0]    

    return null



def plot_nulls(null_count):

    f, ax = plt.subplots(figsize=(15, 6))

    plt.xticks(rotation='90')

    sns.barplot(x=null_count.index, y=null_count)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15);
# Plot the missing features. 



null_count = nulls(features)

plot_nulls(null_count)

print(f"{len(null_count)} null features")
print(features.PoolQC.dtype)

print("Unique vals: ", features.PoolQC.unique())

print(features.PoolQC.value_counts())
# So most of our observations is already null, we fill those with string "None"



features["PoolQC"] = features["PoolQC"].fillna("None")
# Garage Columns(Features)



garage_cols = []

for col in features.columns:

    if col.startswith("Garage"):

        print(f"{col} : {null_count[col]}")

        garage_cols.append(col)
features.GarageCars.value_counts()
features.GarageQual.value_counts()
features.GarageCond.value_counts()
features.GarageType.value_counts()
# For int dtypes we fill the NaN values with 0s and for the object types,

# we fill with the 'None's.



for col in garage_cols:

    print(f"{col} : {null_count[col]} : {features[col].dtype}")

    

for col in ["GarageYrBlt", "GarageCars", "GarageArea"]:

    features[col].fillna(0, inplace=True)

    

for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:

    features[col].fillna("None", inplace=True)
# We filled some null features with just 0's or 'None's



null_count = nulls(features)

plot_nulls(null_count)

print(f"{len(null_count)} null features")
# Only 1 observation has null value, so we fill that observation with most

# frequent element which is "SBrkr"



print(features.Electrical.value_counts().sum(), " - ", len(features))

print(null_count.Electrical)

print(features.Electrical.value_counts())



features.Electrical.fillna("SBrkr", inplace=True)
# Only 1 observation has null value, so we fill that observation with most

# frequent element which is "WD"



print(features.SaleType.value_counts().sum(), " - ", len(features))

print(null_count.SaleType)

print(features.SaleType.value_counts())



features.SaleType.fillna("WD", inplace=True)
# Only 1 observation has null value, so we fill that observation with most

# frequent element which is "VinylSd"



print(features.Exterior1st.value_counts().sum(), " - ", len(features))

print(null_count.Exterior1st)

print(features.Exterior1st.value_counts())



features.Exterior1st.fillna("VinylSd", inplace=True)
# Only 1 observation has null value, so we fill that observation with most

# frequent element which is "VinylSd"



print(features.Exterior2nd.value_counts().sum(), " - ", len(features))

print(null_count.Exterior2nd)

print(features.Exterior2nd.value_counts())



features.Exterior2nd.fillna("VinylSd", inplace=True)
# Only 1 observation has null value, so we fill that observation with most

# frequent element which is "TA"



print(features.KitchenQual.value_counts().sum(), " - ", len(features))

print(null_count.KitchenQual)

print(features.KitchenQual.value_counts())



features.KitchenQual.fillna("TA", inplace=True)
# Only 2 observation has null value, so we fill that observation with most

# frequent element which is "Typ"



print(features.Functional.value_counts().sum(), " - ", len(features))

print(null_count.Functional)

print(features.Functional.value_counts())



features.Functional.fillna("Typ", inplace=True)
null_count = nulls(features)

plot_nulls(null_count)

print(f"{len(null_count)} null features")
null_count[null_count < 0.1]
# Those are categoric columns. So we fill them with the "None". 



cat_columns = ["BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType2", "BsmtFinType1"]

for col in cat_columns:

    features[col].fillna("None", inplace=True)
null_count = nulls(features)

plot_nulls(null_count)

print(f"{len(null_count)} null features")
features[features.MSZoning.isnull()]
# groupby MSSubClass and find the most frequent MSZoning for that particular MSSubClass. 

# Then fill the MSZoning with that element. 



features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# For the rest of the object/categoric columns we fill the null values with "None".



for col in categoric_columns:

    features[col].fillna("None", inplace=True)



# For the numeric columns, we fill with 0s.



for col in numeric_columns:

    features[col].fillna(0, inplace=True)
pd.get_dummies(features.MSZoning)[:10]
print(features.shape)

final_features = pd.get_dummies(features).reset_index(drop=True)

print(final_features.shape)
X = final_features.iloc[:len(y), :]

test_X = final_features.iloc[len(X):, :]



print('X: ', X.shape)

print('y: ', y.shape)

print('test_X: ', test_X.shape)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X, y)
# This is from fastai library.

# https://github.com/fastai



def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)
fi = rf_feat_importance(m, X); fi[:10]
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)



plot_fi(fi[:30]);
# Total size of the house in square feet. 

features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



# This is the most basic type of feature engineering. 

# There are also BsmtUnfSF, LowQualFinSF features as well to 

# add additional information. 

features['TotalSQFootage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



# Total number of bathrooms.

features['TotalBathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



# Total size of the porch in square feet.

features['TotalPorchSF'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])
print(features.shape)

final_features = pd.get_dummies(features).reset_index(drop=True)

print(final_features.shape)



X = final_features.iloc[:len(y), :]

test_X = final_features.iloc[len(X):, :]



print('X: ', X.shape)

print('y: ', y.shape)

print('test_X: ', test_X.shape)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X, y)
fi = rf_feat_importance(m, X); fi[:10]
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
plot_fi(fi[:30]);
# After filling nulls with bunch of 0's and 'None's there are no more nulls.



null_count = nulls(features)

null_count
# Current columns



features.columns
# We are going to drop some features that we used above to create new ones.



features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1',

               'BsmtFinSF2', '1stFlrSF', 'FullBath', 'GarageArea', 

               'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF',

               '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF', 'PoolArea',

               'TotalBsmtSF', 'Fireplaces'], inplace=True, axis=1)
# This is from fastai's machine learning course. 

# http://course18.fast.ai/ml



from scipy.cluster import hierarchy as hc



corr = np.round(scipy.stats.spearmanr(features).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16, 20))

dendrogram = hc.dendrogram(z, labels=features.columns, orientation='left', leaf_font_size=16)

plt.show()
print(features.shape)

final_features = pd.get_dummies(features).reset_index(drop=True)

print(final_features.shape)
X = final_features.iloc[:len(y), :]

test_X = final_features.iloc[len(X):, :]



print('X: ', X.shape)

print('y: ', y.shape)

print('test_X: ', test_X.shape)
def rmse(preds, labels):

    return np.sqrt(mean_squared_error(preds, labels))



def train_error(estimator, X_train, y_train):

    preds = estimator.predict(X_train)

    print("RMSE: ", rmse(preds, y_train))
elastic_net = ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10],

                                    l1_ratio=[.01, .1, .5, .9, .99],

                                    max_iter=5000).fit(X, y)

train_error(elastic_net, X, y)
en_model = elastic_net.fit(X, y)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

# Hyperparameter options



gbr_hyperparams = {'n_estimators': [1000, 3000],

                   'learning_rate': [0.05],

                   'max_depth': [3],

                   'max_features': ['sqrt', 0.5],

                   'min_samples_leaf': [10, 15],

                   'min_samples_split': [5, 10],

                   'loss': ['huber']}



model_gbr = GradientBoostingRegressor()



gs = GridSearchCV(model_gbr, gbr_hyperparams)

gs.fit(X, y)

# print(gs.best_params_)
gs.best_params_
best_gbr = gs.best_estimator_

best_gbr
# https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMRegressor

# https://lightgbm.readthedocs.io/en/latest/Parameters.html

# Hyperparameter options



lgb_hyperparams = {'num_leaves': [5, 10], 

               'learning_rate': [0.05],

               'n_estimators': [500, 1000],

               'bagging_fraction': [0.8],

               'bagging_freq': [3, 5],

               'feature_fraction': [0.6, 0.8],

               'feature_fraction_seed': [9],

               'bagging_seed': [9],

               'min_data_in_leaf': [5, 10]

              }



model_lgb = LGBMRegressor(objective='regression', n_jobs=-1)



gs = GridSearchCV(model_lgb, lgb_hyperparams)

gs.fit(X, y)
best_lgb = gs.best_estimator_

best_lgb
# https://xgboost.readthedocs.io/en/latest/parameter.html

# Hyperparameter options



xgb_hyperparams = {'colsample_bytree': [0.4, 0.5],

                   'gamma': [0.04],

                   'learning_rate': [0.05],

                   'max_depth': [3, 5],

                   'n_estimators': [1000, 2000],

                   'reg_alpha': [0.4, 0.6],

                   'reg_lambda': [0.6, 0.8],

                   'subsample': [0.5, 0.8]

                   }





model_xgb = XGBRegressor(random_state=7)



gs = GridSearchCV(model_xgb, xgb_hyperparams)

gs.fit(X, y)
best_xgb = gs.best_estimator_

best_xgb
best_xgb = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,

             learning_rate=0.05, loss='huber', max_depth=3,

             max_features='sqrt', max_leaf_nodes=None,

             min_impurity_decrease=0.0, min_impurity_split=None,

             min_samples_leaf=10, min_samples_split=5,

             min_weight_fraction_leaf=0.0, n_estimators=3000,

             presort='auto', random_state=None, subsample=1.0, verbose=0,

             warm_start=False).fit(X, y)

             

             

best_lgb = LGBMRegressor(bagging_fraction=0.8, bagging_freq=3, bagging_seed=9,

       boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

       feature_fraction=0.6, feature_fraction_seed=9,

       importance_type='split', learning_rate=0.05, max_depth=-1,

       min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=5,

       min_split_gain=0.0, n_estimators=1000, n_jobs=-1, num_leaves=5,

       objective='regression', random_state=None, reg_alpha=0.0,

       reg_lambda=0.0, silent=True, subsample=1.0,

       subsample_for_bin=200000, subsample_freq=0).fit(X, y)

       

       

best_gbr = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.4, gamma=0.04, importance_type='gain',

       learning_rate=0.05, max_delta_step=0, max_depth=3,

       min_child_weight=1, missing=None, n_estimators=2000, n_jobs=1,

       nthread=None, objective='reg:linear', random_state=7, reg_alpha=0.6,

       reg_lambda=0.6, scale_pos_weight=1, seed=None, silent=True,

       subsample=0.8).fit(X, y)
train_error(best_gbr, X, y)
train_error(best_lgb, X, y)
train_error(best_xgb, X, y)
# Stacking models

# Here we exponentiate the predictions to transform original version.



preds = (np.exp(best_xgb.predict(test_X)) + 

                np.exp(en_model.predict(test_X)) + 

                np.exp(best_lgb.predict(test_X)) + 

                np.exp(best_gbr.predict(test_X))) / 4
# 0.11865



submission = pd.DataFrame({'Id': test_ID, 'SalePrice': preds})

submission.to_csv('submission.csv', index =False)