import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder





def xgboost_model(X_train, X_valid, y_train, y_valid, X_test, file_name): 

    print('XGBoost')

    xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05)

    xgb_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)

    y_predict = xgb_model.predict(X_valid)

    y_test = xgb_model.predict(X_test)

#     plt.figure()

#     sns.distplot(y_test)

#     print('Average Expected Price =', y_test.mean())

#     print('Standard Deviation =', y_test.std())

    output = pd.DataFrame({'Id':X_test.index, 'SalePrice': y_test})

    output.to_csv(file_name, index=False)

    return int(mean_absolute_error(y_valid, y_predict))



from sklearn.ensemble import RandomForestRegressor

def random_forest_model(X_train, X_valid, y_train, y_valid, X_test, file_name): 

    rf_model = RandomForestRegressor(n_estimators=200, random_state=1)

    rf_model.fit(X_train, y_train)

    y_predict = rf_model.predict(X_valid)

    y_test = rf_model.predict(X_test)

    output = pd.DataFrame({'Id':X_test.index, 'SalePrice': y_test})

    output.to_csv(file_name, index=False)

    return int(mean_absolute_error(y_valid, y_predict))



from catboost import CatBoostRegressor

def catboost_model(X_train, X_valid, y_train, y_valid, X_test, file_name): 

    cb_model = CatBoostRegressor( 

        n_estimators = 200,

        loss_function = 'MAE',

        eval_metric = 'RMSE')

    cb_model.fit( X_train, y_train, use_best_model=True, eval_set=(X_valid, y_valid), silent=True, plot=True )

    y_predict = cb_model.predict(X_valid)

    y_test = cb_model.predict(X_test)

    output = pd.DataFrame({'Id':X_test.index, 'SalePrice': y_test})

    output.to_csv(file_name, index=False)

    return int(mean_absolute_error(y_valid, y_predict))



def check_correlaton(feat_data, label_data):

    fig, axes = plt.subplots(2, 1, figsize=(20, 20))

    for i, corr_method in enumerate(['pearson', 'spearman']):

        coe = feat_data.join(label_data).corr(method=corr_method)

        highly_corr = [col for col in coe.columns if (coe[col].abs()>0.8).sum()>1] # to hide columns correlation to themselves

        high_coe = coe.loc[highly_corr, highly_corr].round(2)

        sns.heatmap(high_coe, mask= np.triu(high_coe), square=True, annot=True, annot_kws={"size": 10}, cmap="BuPu", ax=axes[i])

        axes[i].set_title(corr_method, fontsize=18)

    fig.tight_layout(pad=3.0)

    

def k_fold_xgboost(X, y, X_test, file_name, k_folds):

    print(f'XGBoost With {k_folds} Cross-Validation')

    result = None

    total_error = 0

    valid_size = len(X)/k_folds

    for kf in range(k_folds):

        valid_start = int(kf*valid_size)

        valid_end = int(valid_start + valid_size)

        X_valid = X.iloc[valid_start:valid_end]

        y_valid = y.iloc[valid_start:valid_end]

        X_train = X.drop(X[valid_start:valid_end].index)

        y_train = y.drop(y[valid_start:valid_end].index)

        xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05)

        xgb_model.fit(X_train, y_train, 

                 early_stopping_rounds=5, 

                 eval_set=[(X_valid, y_valid)], 

                 verbose=False)

        y_predict = xgb_model.predict(X_valid)

        total_error += mean_absolute_error(y_valid, y_predict)

        y_test = xgb_model.predict(X_test)

        if result is None:

            result = y_test

        else:

            result += y_test

    final_result = result / k_folds

#     plt.figure()

#     sns.distplot(final_result)

#     print('Average Expected Price =', final_result.mean())

#     print('Standard Deviation =', final_result.std())

    output = pd.DataFrame({'Id':X_test.index, 'SalePrice': final_result})

    output.to_csv(file_name, index=False)

    return int(total_error/k_folds)

train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

X_test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)



train.dropna(subset=['SalePrice'], axis=0, inplace=True)

y = train['SalePrice']

X = train.drop(['SalePrice'], axis=1)



num_features = list(X.select_dtypes(exclude=['object']).columns)

num_features.remove('MSSubClass')



cat_features = list(X.select_dtypes(include=['object']).columns)

cat_features.append('MSSubClass')



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)



# Missing Data



num_imputer = SimpleImputer(strategy='mean')

X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[num_features]))

X_valid_num = pd.DataFrame(num_imputer.transform(X_valid[num_features]))

X_test_num = pd.DataFrame(num_imputer.transform(X_test[num_features]))

X_train_num.columns = X_train[num_features].columns

X_valid_num.columns = X_valid[num_features].columns

X_test_num.columns = X_test[num_features].columns

X_test_num.index = X_test[num_features].index



cat_imputer = SimpleImputer(strategy='most_frequent')

X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_features]))

X_valid_cat = pd.DataFrame(cat_imputer.transform(X_valid[cat_features]))

X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[cat_features]))

X_train_cat.columns = X_train[cat_features].columns

X_valid_cat.columns = X_valid[cat_features].columns

X_test_cat.columns = X_test[cat_features].columns

X_test_cat.index = X_test[cat_features].index





# Cat to Num conversion



oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

X_train_encoded = pd.DataFrame(oh_encoder.fit_transform(X_train_cat))

X_valid_encoded = pd.DataFrame(oh_encoder.transform(X_valid_cat))

X_test_encoded = pd.DataFrame(oh_encoder.transform(X_test_cat))

X_train_encoded.index = X_train_cat.index

X_valid_encoded.index= X_valid_cat.index

X_test_encoded.index= X_test_cat.index



# put columns back together



X_train_modified = X_train_num.join(X_train_encoded)

X_valid_modified = X_valid_num.join(X_valid_encoded)

X_test_modified = X_test_num.join(X_test_encoded)



# check model score



print(xgb_model(X_train_modified, X_valid_modified, y_train, y_valid, X_test_modified, 'model_1.csv'))



print('Competition Score for this model is: 15170.35693')
train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

X_test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)



train.dropna(subset=['SalePrice'], axis=0, inplace=True)

y = train['SalePrice']

X = train.drop(['SalePrice'], axis=1)



# # Solve I/O leakage

# leaked_features = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition']

# X.drop(leaked_features, axis=1, inplace=True)

# X_test.drop(leaked_features, axis=1, inplace=True)



num_features = list(X.select_dtypes(exclude=['object']).columns)

num_features.remove('MSSubClass')



cat_features = list(X.select_dtypes(include=['object']).columns)

cat_features.append('MSSubClass')





# Numerical features

disc_features = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

                     'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

cont_features = [f for f in num_features if f not in disc_features]



# Categorical features

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley',

                       ]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }

nominal_features = [f for f in cat_features if f not in ordinal_features]





# for ordinal features, replace all Nan values with 'NA' ( which will be later converted to zero)

for col in ordinal_features:

    X[col].fillna(value='NA', inplace=True)

    X_test[col].fillna(value='NA', inplace=True)



# for features to be one-hot encoded, will make all 'None' and 'NA' be Nan

for col in nominal_features:

    X[col].replace(['None', 'NA'], pd.NA, inplace=True)

    X_test[col].replace(['None', 'NA'], pd.NA, inplace=True)



# Nan values that should be filled with zeros

features_nan_zero = ['MasVnrArea']

for col in features_nan_zero:

    X[col].fillna(0, inplace=True)

    X_test[col].fillna(0, inplace=True)



# Nan filled with mean

features_nan_mean = ['LotFrontage', 'GarageYrBlt']

for col in features_nan_mean:

    X[col].fillna(X[col].mean(), inplace=True)

    X_test[col].fillna(X_test[col].mean(), inplace=True)



for col in ordinal_features:

    X[col] = X[col].map(ordinal_mapping)

    X_test[col] = X_test[col].map(ordinal_mapping)



# covert remaining categorical features (one-hot-features) to number

X_full = pd.concat([X[nominal_features], X_test[nominal_features]])

X_full_dummies = pd.get_dummies(X_full[nominal_features], dtype=int)

X_dummies = X_full_dummies.loc[0:1460].copy()

X_test_dummies = X_full_dummies.loc[1460:].copy()

X.drop(nominal_features, axis=1, inplace=True)

X_test.drop(nominal_features, axis=1, inplace=True)

X = X.join(X_dummies)

X_test = X_test.join(X_test_dummies)



# check model score

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

print(xgb_model(X_train, X_valid, y_train, y_valid, X_test, 'model_2.csv'))



# # na_count = X.isnull().sum()

# # na_count[na_count > 0].sort_values(ascending=False)



print('Competition Score for this model is: 15057.64597')
train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

X_test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)



train.dropna(subset=['SalePrice'], axis=0, inplace=True)

y = train['SalePrice']

X = train.drop(['SalePrice'], axis=1)



# # Solve I/O leakage

# leaked_features = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition']

# X.drop(leaked_features, axis=1, inplace=True)

# X_test.drop(leaked_features, axis=1, inplace=True)



num_features = list(X.select_dtypes(exclude=['object']).columns)

num_features.remove('MSSubClass')



cat_features = list(X.select_dtypes(include=['object']).columns)

cat_features.append('MSSubClass')





# Numerical features

disc_features = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

                     'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

cont_features = [f for f in num_features if f not in disc_features]



# Categorical features

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley',

                       ]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }

nominal_features = [f for f in cat_features if f not in ordinal_features]





# for ordinal features, replace all Nan values with 'NA' ( which will be later converted to zero)

for col in ordinal_features:

    X[col].fillna(value='NA', inplace=True)

    X_test[col].fillna(value='NA', inplace=True)



# for features to be one-hot encoded, will make all 'None' and 'NA' be Nan

for col in nominal_features:

    X[col].replace(['None', 'NA'], pd.NA, inplace=True)

    X_test[col].replace(['None', 'NA'], pd.NA, inplace=True)



# here nan is because there is no data

features_nan_zero = ['MasVnrArea', 'GarageYrBlt']

for col in features_nan_zero:

    X[col].fillna(0, inplace=True)

    X_test[col].fillna(0, inplace=True)



# there should be data, but was not recorded

X['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)

X_test['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)



for col in ordinal_features:

    X[col] = X[col].map(ordinal_mapping)

    X_test[col] = X_test[col].map(ordinal_mapping)



# covert remaining categorical features (one-hot-features) to number

X_full = pd.concat([X[nominal_features], X_test[nominal_features]])

X_full_dummies = pd.get_dummies(X_full[nominal_features], dtype=int)

X_dummies = X_full_dummies.loc[0:1460].copy()

X_test_dummies = X_full_dummies.loc[1460:].copy()

X.drop(nominal_features, axis=1, inplace=True)

X_test.drop(nominal_features, axis=1, inplace=True)

X = X.join(X_dummies)

X_test = X_test.join(X_test_dummies)



# check model score

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

print(xgb_model(X_train, X_valid, y_train, y_valid, X_test, 'model_3.csv'))



# # na_count = X.isnull().sum()

# # na_count[na_count > 0].sort_values(ascending=False)





print('Competition Score for this model is: 15039.19075')
train_data = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

test_data = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)



train_data.dropna(subset=['SalePrice'], axis=0, inplace=True)

y = train_data['SalePrice']

train_data.drop(['SalePrice'], axis=1, inplace=True)



X_concat = pd.concat([train_data, test_data])



num_features = list(X_concat.select_dtypes(exclude=['object']).columns)

num_features.remove('MSSubClass')



cat_features = list(X_concat.select_dtypes(include=['object']).columns)

cat_features.append('MSSubClass')





# Numerical features

disc_features = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

                     'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

cont_features = [f for f in num_features if f not in disc_features]



# Categorical features

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley',

                       ]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }

nominal_features = [f for f in cat_features if f not in ordinal_features]





# for ordinal features, replace all Nan values with 'NA' ( which will be later converted to zero)

for col in ordinal_features:

    X_concat[col].fillna(value='NA', inplace=True)



# for features to be one-hot encoded, will make all 'None' and 'NA' be Nan

for col in nominal_features:

    X_concat[col].replace(['None', 'NA'], pd.NA, inplace=True)



# here nan is because there is no data

for col in ['MasVnrArea', 'GarageYrBlt']:

    X_concat[col].fillna(0, inplace=True)



# there should be data, but was not recorded

X_concat['LotFrontage'].fillna(X_concat['LotFrontage'].mean(), inplace=True)



for col in ordinal_features:

    X_concat[col] = X_concat[col].map(ordinal_mapping)



# covert remaining categorical features (one-hot-features) to number

X_dummies = pd.get_dummies(X_concat[nominal_features], dtype=int)

X_concat.drop(nominal_features, axis=1, inplace=True)

X_concat = X_concat.join(X_dummies)

X = X_concat.loc[0:1460].copy()

X_test = X_concat.loc[1461:].copy()



# check model score

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

print(xgb_model(X_train, X_valid, y_train, y_valid, X_test, 'model_4.csv'))



print('Competition Score for this model is: 15266.01175')
train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

X_test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)



train.dropna(subset=['SalePrice'], axis=0, inplace=True)

y = train['SalePrice']

X = train.drop(['SalePrice'], axis=1)



num_features = list(X.select_dtypes(exclude=['object']).columns)

num_features.remove('MSSubClass')



cat_features = list(X.select_dtypes(include=['object']).columns)

cat_features.append('MSSubClass')



# Numerical features

disc_features = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

                     'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

cont_features = [f for f in num_features if f not in disc_features]



# Categorical features

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley',

                       ]

nominal_features = [f for f in cat_features if f not in ordinal_features]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }





# for ordinal features, replace all Nan values with 'NA' ( which will be later converted to zero)

X[ordinal_features].apply(lambda x: x.fillna(value='NA', inplace=True))

X_test[ordinal_features].apply(lambda x: x.fillna(value='NA', inplace=True))



# for features to be one-hot encoded, will make all 'None' and 'NA' be Nan

X[nominal_features].apply(lambda x: x.replace(['None', 'NA'], pd.NA, inplace=True))

X_test[nominal_features].apply(lambda x: x.replace(['None', 'NA'], pd.NA, inplace=True))



# here nan is because there is no data

features_nan_zero = ['MasVnrArea', 'GarageYrBlt']

X[features_nan_zero].apply(lambda x: x.fillna(0, inplace=True))

X_test[features_nan_zero].apply(lambda x: x.fillna(0, inplace=True))



# there should be data, but was not recorded

X['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)

X_test['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)



X[ordinal_features] = X[ordinal_features].apply(lambda x: x.map(ordinal_mapping))

X_test[ordinal_features] = X_test[ordinal_features].apply(lambda x: x.map(ordinal_mapping))



# covert remaining categorical features (one-hot-features) to number

X_full = pd.concat([X[nominal_features], X_test[nominal_features]])

X_full_dummies = pd.get_dummies(X_full[nominal_features])

X_dummies = X_full_dummies.loc[0:1460].copy()

X_test_dummies = X_full_dummies.loc[1460:].copy()

X.drop(nominal_features, axis=1, inplace=True)

X_test.drop(nominal_features, axis=1, inplace=True)

X = X.join(X_dummies)

X_test = X_test.join(X_test_dummies)



# check model score

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

print(xgb_model(X_train, X_valid, y_train, y_valid, X_test, 'model_5.csv'))



print('Competition Score for this model is: 14909.98007')
train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

X_test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)



train.dropna(subset=['SalePrice'], axis=0, inplace=True)

y = train['SalePrice']

X = train.drop(['SalePrice'], axis=1)



num_features = list(X.select_dtypes(exclude=['object']).columns)

num_features.remove('MSSubClass')



cat_features = list(X.select_dtypes(include=['object']).columns)

cat_features.append('MSSubClass')



# Numerical features

disc_features = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

                     'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

cont_features = [f for f in num_features if f not in disc_features]



# Categorical features

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley',

                       ]

nominal_features = [f for f in cat_features if f not in ordinal_features]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }



for df in [X, X_test]:

    # fixing na issue

    df[ordinal_features].apply(lambda x: x.fillna(value='NA', inplace=True))

    df[nominal_features].apply(lambda x: x.replace(['None', 'NA'], pd.NA, inplace=True))

    df[['MasVnrArea', 'GarageYrBlt']].apply(lambda x: x.fillna(0, inplace=True))

    df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)

    df[ordinal_features] = df[ordinal_features].apply(lambda x: x.map(ordinal_mapping))

    # generating new features

    df['total_surface_area'] = df['1stFlrSF'] + df['2ndFlrSF']

    df['total_baths'] = df['FullBath'] + df['HalfBath']

    df['total_bsmt_finish'] = df['BsmtFinSF1'] + df['BsmtFinSF2']

    df['total_bsmt_baths'] = df['BsmtFullBath'] + df['BsmtHalfBath']

    df['number_sold_same_month'] = df.groupby(['Neighborhood', "YrSold", 'MoSold'])['MoSold'].transform('count')

#     df['rolling'] = df.groupby(['Neighborhood', "YrSold", 'MoSold','number_sold_same_month'])['number_sold_same_month'].transform(lambda x: x.rolling(3).sum())

#     print(df.sort_values(by=['Neighborhood', "YrSold", 'MoSold']).groupby(['Neighborhood', "YrSold", 'MoSold','number_sold_same_month'])[['Neighborhood', "YrSold", 'MoSold','number_sold_same_month','rolling']].head(10))

#     for key, value in df.groupby(['Neighborhood', "YrSold", 'MoSold']):

#         print(key.rolling(3))

#         print(value[['number_sold_same_month','MoSold']])

#     print(df.groupby(['Neighborhood', "YrSold"])[['MoSold','number_sold_same_month']].rolling(3).sum())

#     print(df[['number_sold_same_month', "YrSold", 'MoSold']].sort_values(by=['MoSold']))



# covert remaining categorical features (one-hot-features) to number

X_full = pd.concat([X[nominal_features], X_test[nominal_features]])

X_full_dummies = pd.get_dummies(X_full[nominal_features])

X_dummies = X_full_dummies.loc[0:1460].copy()

X_test_dummies = X_full_dummies.loc[1460:].copy()

X.drop(nominal_features, axis=1, inplace=True)

X_test.drop(nominal_features, axis=1, inplace=True)

X = X.join(X_dummies)

X_test = X_test.join(X_test_dummies)



# check model score

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

print(xgb_model(X_train, X_valid, y_train, y_valid, X_test, 'model_6.csv'))



print('Competition Score for this model is: 14597.31148')
train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

X_test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)



train.dropna(subset=['SalePrice'], axis=0, inplace=True)

y = train['SalePrice'].copy()

X = train.drop(['SalePrice'], axis=1).copy()



num_features = list(X.select_dtypes(exclude=['object']).columns)

num_features.remove('MSSubClass')



cat_features = list(X.select_dtypes(include=['object']).columns)

cat_features.append('MSSubClass')



# Numerical features

disc_features = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

                     'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

cont_features = [f for f in num_features if f not in disc_features]



# Categorical features

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley',

                       ]

nominal_features = [f for f in cat_features if f not in ordinal_features]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }



# plt.figure(figsize=(14,12))

# correlation = train[num_features].corr()

# sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.5, cmap='Blues')

# print(list(train[num_features]))



for df in [X, X_test]:

    # fixing na issue

    df[ordinal_features].apply(lambda x: x.fillna(value='NA', inplace=True))

    df[nominal_features].apply(lambda x: x.replace(['None', 'NA'], pd.NA, inplace=True))

    df[['MasVnrArea', 'GarageYrBlt']].apply(lambda x: x.fillna(0, inplace=True))

    df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)

    df[ordinal_features] = df[ordinal_features].apply(lambda x: x.map(ordinal_mapping))

    # generating new features

    df['total_surface_area'] = df['1stFlrSF'] + df['2ndFlrSF']

    df['total_baths'] = df['FullBath'] + df['HalfBath']

    df['total_bsmt_finish'] = df['BsmtFinSF1'] + df['BsmtFinSF2']

    df['total_bsmt_baths'] = df['BsmtFullBath'] + df['BsmtHalfBath']

    df['number_sold_same_month'] = df.groupby(['Neighborhood', "YrSold", 'MoSold'])['MoSold'].transform('count')

    

    df['Proch_Deck'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch'] +df['WoodDeckSF']

    df['Proch_Deck_bin'] = df['Proch_Deck'].apply(lambda x: 1 if x > 0 else 0)

    df.drop(['Proch_Deck'], axis=1, inplace=True)



# covert remaining categorical features (one-hot-features) to number

X_full = pd.concat([X[nominal_features], X_test[nominal_features]])

X_full_dummies = pd.get_dummies(X_full[nominal_features])

X_dummies = X_full_dummies.loc[0:1460].copy()

X_test_dummies = X_full_dummies.loc[1460:].copy()

X.drop(nominal_features, axis=1, inplace=True)

X_test.drop(nominal_features, axis=1, inplace=True)

X = X.join(X_dummies)

X_test = X_test.join(X_test_dummies)



# check model score

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

print(xgb_model(X_train, X_valid, y_train, y_valid, X_test, 'model_7.csv'))



print('Competition Score for this model is: 14560.28803')
train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

X_test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)



train.dropna(subset=['SalePrice'], axis=0, inplace=True)

y = train['SalePrice'].copy()

X = train.drop(['SalePrice'], axis=1).copy()



num_features = list(X.select_dtypes(exclude=['object']).columns)

cat_features = list(X.select_dtypes(include=['object']).columns)



# Numerical features

disc_features = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

                     'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

cont_features = [f for f in num_features if f not in disc_features]



# Categorical features

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley'

                       ]

nominal_features = [f for f in cat_features if f not in ordinal_features]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }



X_test.iloc[1132,58] = 2007



for df in [X, X_test]:

    # fillna for numerical features

    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

    for feat in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 

                 'MasVnrArea', 'GarageYrBlt', 'GarageCars', 'GarageArea']:

        df[feat] = df[feat].fillna(0)

    # fillna for categorical features

    for feat in ['MSZoning', 'Utilities', 'Exterior1st', 'Electrical']:

        df[feat] = df.groupby('MSSubClass')[feat].transform(lambda x: x.fillna(x.mode()[0]))

    for feat in ['KitchenQual', 'Functional', 'SaleType']:

        df[feat].fillna(df[feat].mode()[0], inplace=True)

    df[ordinal_features] = df[ordinal_features].apply(lambda x: x.fillna(value='NA'))

    # convert ordinal features to numerical features

    df[ordinal_features] = df[ordinal_features].apply(lambda x: x.map(ordinal_mapping))

    # generating new features

    df['total_baths'] = df['FullBath'] + df['HalfBath']

    df['total_bsmt_finish'] = df['BsmtFinSF1'] + df['BsmtFinSF2']

    df['total_bsmt_baths'] = df['BsmtFullBath'] + df['BsmtHalfBath']

    df['proch_deck_area'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

    df['proch_deck_binary'] = df['proch_deck_area'].apply(lambda x: 1 if x > 0 else 0)

    df['sold_same_month'] = df.groupby(['Neighborhood', "YrSold", 'MoSold'])['MoSold'].transform('count')

    df['num_months_passed'] = (df['YrSold'] - 2006)*12 + df['MoSold']



# covert remaining categorical features (one-hot-features) to number

X_full = pd.concat([X[nominal_features], X_test[nominal_features]])

X_full_dummies = pd.get_dummies(X_full[nominal_features])

X_dummies = X_full_dummies.loc[0:1460].copy()

X_test_dummies = X_full_dummies.loc[1460:].copy()

X.drop(nominal_features, axis=1, inplace=True)

X_test.drop(nominal_features, axis=1, inplace=True)

X = X.join(X_dummies)

X_test = X_test.join(X_test_dummies)



# check model score

print(f"Validation Score: {k_fold_xgboost(X, y, X_test, 'model_8_with_cross_val.csv',3)}\n")

# X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

# print('Validation Score', xgboost_model(X_train, X_valid, y_train, y_valid, X_test, 'model_8.csv'))

train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)

train.dropna(subset=['SalePrice'], axis=0, inplace=True)

y = train['SalePrice']

train.drop(['SalePrice'], axis=1, inplace=True)

train_test = pd.concat([train, test])



# Distinguishing features

train_test['MSSubClass'] = train_test['MSSubClass'].astype(str)

num_features = list(train_test.select_dtypes(exclude=['object']).columns)

cat_features = list(train_test.select_dtypes(include=['object']).columns)

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley',

                       ]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }

nominal_features = [f for f in cat_features if f not in ordinal_features]



# correcting wrong values

train_test.iloc[2592,58] = 2007



# missing values in numerical features

train_test['LotFrontage'] = train_test.groupby('MSSubClass')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

for feat in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea', 'GarageYrBlt', 'GarageCars', 'GarageArea']:

    train_test[feat].fillna(0, inplace=True)

    

# missing values in categorical features

for feat in ['MSZoning', 'Utilities', 'Exterior1st', 'Electrical']:

    train_test[feat] = train_test.groupby('MSSubClass')[feat].transform(lambda x: x.fillna(x.mode()[0]))

for feat in ['KitchenQual', 'Functional', 'SaleType']:

    train_test[feat].fillna(train_test[feat].mode()[0], inplace=True)

train_test[ordinal_features] = train_test[ordinal_features].apply(lambda x: x.fillna('NA'))



# convert ordinal features to numbers

train_test[ordinal_features] = train_test[ordinal_features].apply(lambda x: x.map(ordinal_mapping))



# convert nominal features to numbers

dummies = pd.get_dummies(train_test[nominal_features])

train_test = train_test.join(dummies)

                                                                                        

# generating new features

train_test['total_surface_area'] = train_test['1stFlrSF'] + train_test['2ndFlrSF']

train_test['total_baths'] = train_test['FullBath'] + train_test['HalfBath']

train_test['total_bsmt_finish'] = train_test['BsmtFinSF1'] + train_test['BsmtFinSF2']

train_test['total_bsmt_baths'] = train_test['BsmtFullBath'] + train_test['BsmtHalfBath']

train_test['proch_deck_area'] = train_test['OpenPorchSF'] + train_test['EnclosedPorch'] + train_test['ScreenPorch'] +train_test['WoodDeckSF']

train_test['proch_deck_binary'] = train_test['proch_deck_area'].apply(lambda x: 1 if x > 0 else 0)

train_test['sold_same_month'] = train_test.groupby(['Neighborhood', "YrSold", 'MoSold'])['MoSold'].transform('count')

train_test['num_months_passed'] = (train_test['YrSold'] - 2006)*12 + train_test['MoSold']



# drop unuseful features

train_test.drop(nominal_features, axis=1, inplace=True)



X = train_test.loc[0:1460]

X_test = train_test.loc[1461:]



# check model score

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

print(xgb_model(X_train, X_valid, y_train, y_valid, X_test, 'model_9.csv'))



print('Competition Score for this model is: ')
train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)

train.dropna(subset=['SalePrice'], axis=0, inplace=True)

y = train['SalePrice']

train.drop(['SalePrice'], axis=1, inplace=True)



# Distinguishing features

for df in [train, test]:

    df['MSSubClass'] = df['MSSubClass'].astype(str)

    

num_features = list(train.select_dtypes(exclude=['object']).columns)

cat_features = list(train.select_dtypes(include=['object']).columns)

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley'

                       ]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }

nominal_features = [f for f in cat_features if f not in ordinal_features]



# correcting wrong values

test.iloc[1132,58] = 2007



# missing values in numerical features

for df in [train, test]:

    df['LotFrontage'] = df.groupby('MSSubClass')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

    for feat in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea', 'GarageYrBlt', 'GarageCars', 'GarageArea']:

        df[feat] = df[feat].fillna(0)

    

    # missing values in categorical features

    for feat in ['MSZoning', 'Utilities', 'Exterior1st', 'Electrical']:

        df[feat] = df.groupby('MSSubClass')[feat].transform(lambda x: x.fillna(x.mode()[0]))

    for feat in ['KitchenQual', 'Functional', 'SaleType']:

        df[feat] = df[feat].fillna(df[feat].mode()[0])

    df[ordinal_features] = df[ordinal_features].apply(lambda x: x.fillna('NA'))



    # convert ordinal features to numbers

    df[ordinal_features] = df[ordinal_features].apply(lambda x: x.map(ordinal_mapping))

                                                                                        

    # generating new features

    df['total_surface_area'] = df['1stFlrSF'] + df['2ndFlrSF']

    df['total_baths'] = df['FullBath'] + df['HalfBath']

    df['total_bsmt_finish'] = df['BsmtFinSF1'] + df['BsmtFinSF2']

    df['total_bsmt_baths'] = df['BsmtFullBath'] + df['BsmtHalfBath']

    df['proch_deck_area'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

    df['proch_deck_binary'] = df['proch_deck_area'].apply(lambda x: 1 if x > 0 else 0)

    df['sold_same_month'] = df.groupby(['Neighborhood', "YrSold", 'MoSold'])['MoSold'].transform('count')

    df['num_months_passed'] = (df['YrSold'] - 2006)*12 + df['MoSold']



# convert nominal features to numbers

train_test = pd.concat([train, test])

dummies = pd.get_dummies(train_test[nominal_features])

train_test = train_test.join(dummies)

train_test = train_test.drop(nominal_features, axis=1)

# single_val_cols = [col for col in train_test.columns if train_test[col].value_counts().max()/len(train_test)>0.998]

# train_test.drop(single_val_cols, axis=1, inplace=True)





X = train_test.loc[0:1460]

X_test = train_test.loc[1461:]



# check model score

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

print(xgb_model(X_train, X_valid, y_train, y_valid, X_test, 'model_10.csv'))



print('Competition Score for this model is: ')
train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)

train.dropna(subset=['SalePrice'], axis=0, inplace=True)



y = train['SalePrice']

train.drop(['SalePrice'], axis=1, inplace=True)



# Distinguishing features

for df in [train, test]:

    df['MSSubClass'] = df['MSSubClass'].astype(str)

    

num_features = list(train.select_dtypes(exclude=['object']).columns)

cat_features = list(train.select_dtypes(include=['object']).columns)

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley'

                       ]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }

nominal_features = [f for f in cat_features if f not in ordinal_features]



# correcting wrong values

test.iloc[1132,58] = 2007



# missing values in numerical features

for df in [train, test]:

    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

    for feat in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea', 'GarageYrBlt', 'GarageCars', 'GarageArea']:

        df[feat] = df[feat].fillna(0)

    

    # missing values in categorical features

    for feat in ['MSZoning', 'Utilities', 'Exterior1st', 'Electrical']:

        df[feat] = df.groupby('MSSubClass')[feat].transform(lambda x: x.fillna(x.mode()[0]))

    for feat in ['KitchenQual', 'Functional', 'SaleType']:

        df[feat] = df[feat].fillna(df[feat].mode()[0])

    df[ordinal_features] = df[ordinal_features].apply(lambda x: x.fillna('NA'))



    # convert ordinal features to numbers

    df[ordinal_features] = df[ordinal_features].apply(lambda x: x.map(ordinal_mapping))

                                                                                        

    # generating new features

    df['total_surface_area'] = df['1stFlrSF'] + df['2ndFlrSF']

    df['total_baths'] = df['FullBath'] + df['HalfBath']

    df['total_bsmt_finish'] = df['BsmtFinSF1'] + df['BsmtFinSF2']

    df['total_bsmt_baths'] = df['BsmtFullBath'] + df['BsmtHalfBath']

    df['proch_deck_area'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

    df['proch_deck_binary'] = df['proch_deck_area'].apply(lambda x: 1 if x > 0 else 0)

    df['sold_same_month'] = df.groupby(['Neighborhood', "YrSold", 'MoSold'])['MoSold'].transform('count')

    df['num_months_passed'] = (df['YrSold'] - 2006)*12 + df['MoSold']



# check_correlaton(train, y)

    

# convert nominal features to numbers

train_test = pd.concat([train, test])

dummies = pd.get_dummies(train_test[nominal_features])

train_test = train_test.join(dummies)

train_test = train_test.drop(nominal_features, axis=1)



X = train_test.loc[0:1460]

X_test = train_test.loc[1461:]



# check model score

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

print('XGBoost Score:', xgboost_model(X_train, X_valid, y_train, y_valid, X_test, 'model_11_xgb.csv'))

print('RandomForest Score:', random_forest_model(X_train, X_valid, y_train, y_valid, X_test, 'model_11_rf.csv'))

print('CatBoost Score:', catboost_model(X_train, X_valid, y_train, y_valid, X_test, 'model_11_cb.csv'))





print('Competition best Score for this model is: ')
train = pd.read_csv('../input/homedataformlcourse/train.csv', index_col=0)

test = pd.read_csv('../input/homedataformlcourse/test.csv', index_col=0)

train.dropna(subset=['SalePrice'], axis=0, inplace=True)



y = train['SalePrice']

train.drop(['SalePrice'], axis=1, inplace=True)



# Distinguishing features

for df in [train, test]:

    df['MSSubClass'] = df['MSSubClass'].astype(str)

    

num_features = list(train.select_dtypes(exclude=['object']).columns)

cat_features = list(train.select_dtypes(include=['object']).columns)

ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','BsmtExposure',

                        'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence',

                        'CentralAir', 'Street', 'Alley'

                       ]

ordinal_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0,                                # ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']

                   'Av':2, 'Mn':1, 'No':0,                                                        # ['BsmtExposure']

                   'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1,                                            # ['LotShape']

                   'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1,                                            # ['LandContour']

                   'Pave':2, 'Grvl': 1,                                                           # ['Street', 'Alley']

                   'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1,                                   # ['Utilities']

                   'Gtl':3, 'Mod':2, 'Sev':1,                                                     # ['LandSlope']

                   'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1,                          # ['BsmtFinType1', 'BsmtFinType2']

                   'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1,    # ['Functional']

                   'Fin':3, 'RFn':2, 'Unf':1,                                                     # ['GarageFinish']

                   'Y':3, 'P':2, 'N':1,                                                           # ['PavedDrive']

                   'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1,                                      # ['Fence']

                  }

nominal_features = [f for f in cat_features if f not in ordinal_features]



# correcting wrong values

test.iloc[1132,58] = 2007



# missing values in numerical features

for df in [train, test]:

    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

    for feat in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea', 'GarageYrBlt', 'GarageCars', 'GarageArea']:

        df[feat] = df[feat].fillna(0)

    

    # missing values in categorical features

    for feat in ['MSZoning', 'Utilities', 'Exterior1st', 'Electrical']:

        df[feat] = df.groupby('MSSubClass')[feat].transform(lambda x: x.fillna(x.mode()[0]))

    for feat in ['KitchenQual', 'Functional', 'SaleType']:

        df[feat] = df[feat].fillna(df[feat].mode()[0])

    df[ordinal_features] = df[ordinal_features].apply(lambda x: x.fillna('NA'))



    # convert ordinal features to numbers

    df[ordinal_features] = df[ordinal_features].apply(lambda x: x.map(ordinal_mapping))

                                                                                        

    # generating new features

    df['total_baths'] = df['FullBath'] + df['HalfBath']

    df['total_bsmt_finish'] = df['BsmtFinSF1'] + df['BsmtFinSF2']

    df['total_bsmt_baths'] = df['BsmtFullBath'] + df['BsmtHalfBath']

    df['sold_same_month'] = df.groupby(['Neighborhood', "YrSold", 'MoSold'])['MoSold'].transform('count')

    df['num_months_passed'] = (df['YrSold'] - 2006)*12 + df['MoSold']

    df['proch_deck_area'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['3SsnPorch'] + df['WoodDeckSF']

    df['proch_deck_binary'] = df['proch_deck_area'].apply(lambda x: 1 if x > 0 else 0)

    

    # Drop unuseful columns

    df.drop(['YrSold', 'MoSold', 'proch_deck_area'], axis=1, inplace=True)



# check_correlaton(train, y)

    

# convert nominal features to numbers

train_test = pd.concat([train, test])

dummies = pd.get_dummies(train_test[nominal_features])

train_test = train_test.join(dummies)

train_test = train_test.drop(nominal_features, axis=1)



X = train_test.loc[0:1460]

X_test = train_test.loc[1461:]



# check model score

print('original data Score:', k_fold_xgboost(X, y, X_test, 'model_12.csv',3))

rand_perm = np.random.permutation(len(X))

X_shufl = X.iloc[rand_perm].reset_index().drop(['Id'], axis=1)

y_shufl = y.iloc[rand_perm].reset_index().drop(['Id'], axis=1)

print('shuffled data Score:', k_fold_xgboost(X_shufl, y_shufl, X_test, 'model_12_shuffled.csv',3))
# from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

# from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.svm import SVR

# from sklearn.pipeline import make_pipeline

# from sklearn.preprocessing import RobustScaler

# from sklearn.model_selection import KFold, cross_val_score

# from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

# from mlxtend.regressor import StackingCVRegressor

# from xgboost import XGBRegressor

# from datetime import datetime



# def xval_rmse_scoring(f_model, f_X, f_y, f_cv):

#     """

#     Returns a machine learning model cross-validated score based on the Root Mean Squared Error (RMSE) metric.

    

#     Keyword arguments:

    

#     f_model     Machine learning model.

#                 Object instance

#     f_X_        Tensor containing features for modeling.

#                 Pandas dataframe

#     f_y         Tensor containing targets for modeling.

#                 Pandas series

#     f_cv        Cross-validation splitting strategy.

#                 Please refer to scikit-learn's model_selection cross_val_score for further information.

#     """

#     return np.sqrt(-cross_val_score(f_model, f_X, f_y,

#                                     scoring='neg_mean_squared_error',

#                                     cv=f_cv))









# kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



# elasticnet_alphas = [5e-5, 1e-4, 5e-4, 1e-3]

# elasticnet_l1ratios = [0.8, 0.85, 0.9, 0.95, 1]

# elasticnet = make_pipeline(RobustScaler(),

#                            ElasticNetCV(max_iter=1e7, alphas=elasticnet_alphas,

#                                         cv=kfolds, l1_ratio=elasticnet_l1ratios))



# lasso_alphas = [5e-5, 1e-4, 5e-4, 1e-3]

# lasso = make_pipeline(RobustScaler(),

#                       LassoCV(max_iter=1e7, alphas=lasso_alphas,

#                               random_state=42, cv=kfolds))



# ridge_alphas = [13.5, 14, 14.5, 15, 15.5]

# ridge = make_pipeline(RobustScaler(),

#                       RidgeCV(alphas=ridge_alphas, cv=kfolds))



# gradb = GradientBoostingRegressor(n_estimators=6000, learning_rate=0.01,

#                                   max_depth=4, max_features='sqrt',

#                                   min_samples_leaf=15, min_samples_split=10,

#                                   loss='huber', random_state=42)



# svr = make_pipeline(RobustScaler(),

#                     SVR(C=20, epsilon=0.008, gamma=0.0003))



# xgboost = XGBRegressor(learning_rate=0.01, n_estimators=6000,

#                        max_depth=3, min_child_weight=0,

#                        gamma=0, subsample=0.7,

#                        colsample_bytree=0.7,

#                        objective='reg:squarederror', nthread=-1,

#                        scale_pos_weight=1, seed=27,

#                        reg_alpha=0.00006, random_state=42)



# stackcv = StackingCVRegressor(regressors=(elasticnet, gradb, lasso, 

#                                           ridge, svr, xgboost),

#                               meta_regressor=xgboost,

#                               use_features_in_secondary=True)



# print('Individual model scoring on cross-validation\n')

# print(f'{"Model":<20}{"RMSE mean":>12}{"RMSE stdev":>12}\n')



# score = xval_rmse_scoring(elasticnet, X_train, y_train, kfolds)

# print(f'{"1. ElasticNetCV":<20}{score.mean():>12.4f}{score.std():>12.4f}')



# score = xval_rmse_scoring(lasso, X_train, y_train, kfolds)

# print(f'{"2. LassoCV":<20}{score.mean():>12.4f}{score.std():>12.4f}')



# score = xval_rmse_scoring(ridge, X_train, y_train, kfolds)

# print(f'{"3. RidgeCV":<20}{score.mean():>12.4f}{score.std():>12.4f}')



# score = xval_rmse_scoring(gradb, X_train, y_train, kfolds)

# print(f'{"4. GradientBoosting":<20}{score.mean():>12.4f}{score.std():>12.4f}')



# score = xval_rmse_scoring(svr, X_train, y_train, kfolds)

# print(f'{"5. SVR":<20}{score.mean():>12.4f}{score.std():>12.4f}')



# score = xval_rmse_scoring(xgboost, X_train, y_train, kfolds)

# print(f'{"6. XGBoost":<20}{score.mean():>12.4f}{score.std():>12.4f}')



# print('\nFitting individual models to the training set\n')

# print(f'{"1. ElasticNetCV...":<20}')

# elastic_fit = elasticnet.fit(X_train, y_train)

# print(f'{"2. LassoCV...":<20}')

# lasso_fit = lasso.fit(X_train, y_train)

# print(f'{"3. RidgeCV...":<20}')

# ridge_fit = ridge.fit(X_train, y_train)

# print(f'{"4. GradientBoosting...":<20}')

# gradb_fit = gradb.fit(X_train, y_train)

# print(f'{"5. SVR...":<20}')

# svr_fit = svr.fit(X_train, y_train)

# print(f'{"6. XGBoost...":<20}')

# xgb_fit = xgboost.fit(X_train, y_train)



# print('\nFitting the stacking model to the training set\n')

# print(f'{"StackingCV...":<20}')

# stackcv_fit = stackcv.fit(np.array(X_train), np.array(y_train))



# blend_weights = [0.11, 0.05, 0.00, 0.14, 0.43, 0.00, 0.27]



# y_train = np.expm1(y_train)

# y_pred = np.expm1((blend_weights[0] * elastic_fit.predict(X_train)) +

#                   (blend_weights[1] * lasso_fit.predict(X_train)) +

#                   (blend_weights[2] * ridge_fit.predict(X_train)) +

#                   (blend_weights[3] * svr_fit.predict(X_train)) +

#                   (blend_weights[4] * gradb_fit.predict(X_train)) +

#                   (blend_weights[5] * xgb_fit.predict(X_train)) +

#                   (blend_weights[6] * stackcv_fit.predict(np.array(X_train))))



# rmse = np.sqrt(mean_squared_error(y_train, y_pred))

# rmsle = np.sqrt(mean_squared_log_error(y_train, y_pred))

# mae = mean_absolute_error(y_train, y_pred)

# print('\nBlend model performance on the training set\n')

# print(f'{"RMSE":<7} {rmse:>15.8f}')

# print(f'{"RMSLE":<7} {rmsle:>15.8f}')

# print(f'{"MAE":<7} {mae:>15.8f}')



# print('\nGenerating submission')

# submission = pd.read_csv('submission.csv')

# submission.iloc[:, 1] = np.round_(np.expm1((blend_weights[0] * elastic_fit.predict(X_test)) +

#                                            (blend_weights[1] * lasso_fit.predict(X_test)) +

#                                            (blend_weights[2] * ridge_fit.predict(X_test)) +

#                                            (blend_weights[3] * svr_fit.predict(X_test)) +

#                                            (blend_weights[4] * gradb_fit.predict(X_test)) +

#                                            (blend_weights[5] * xgb_fit.predict(X_test)) +

#                                            (blend_weights[6] * stackcv_fit.predict(np.array(X_test)))))

# submission.to_csv('../output/submission_new.csv', index=False)

# print('Submission saved')
# # (1)

# # Analysis of 'LotArea' feature, and checking for outliers

# feature = 'LotArea'

# print('Mean =', int(train[feature].mean()))

# print('Median =', int(train[feature].median()))

# print('Mode =', int(train[feature].mode()))

# fig, axes = plt.subplots(3, 3, figsize=(16, 12))

# sns.distplot(train[feature].dropna(), ax=axes[0, 0], kde=False)

# sns.boxplot(train[feature].dropna(), ax=axes[0, 1])

# sns.regplot(x=train[feature], y=train['SalePrice'], ax=axes[0, 2])

# sns.scatterplot(x=train[feature], y=train['LotFrontage'], ax=axes[1,0])

# sns.scatterplot(x=train[feature], y=train['TotalBsmtSF'], ax=axes[1,1])

# sns.scatterplot(x=train[feature], y=train['1stFlrSF'], ax=axes[1,2])

# sns.scatterplot(x=train[feature], y=train['GrLivArea'], ax=axes[2,0])

# sns.scatterplot(x=train[feature], y=train['TotRmsAbvGrd'], ax=axes[2,1])

# sns.scatterplot(x=train[feature], y=train['GarageArea'], ax=axes[2,2])

# fig.tight_layout(pad=3.0)



# conclusion, outliers (215245) probably is wrong data, and should be removed.



# # (2)

# feature = 'MoSold' 

# print('Mean =', int(train[feature].mean()))

# print('Median =', int(train[feature].median()))

# print('Mode =', int(train[feature].mode()))

# print('Null values =', train[feature].isnull().sum())

# # print('highest % of any value =', int(train[feature].value_counts()[0]/len(train[feature])*100))

# fig, axes = plt.subplots(2, 3, figsize=(16, 8))

# sns.distplot(train[feature].dropna(), ax=axes[0, 0], kde=False)

# sns.boxplot(train[feature].dropna(), ax=axes[0, 1])

# sns.regplot(x=train[feature], y=train['SalePrice'], ax=axes[0, 2])

# # sns.scatterplot(x=train[feature], y=train['GarageCars'], ax=axes[1,0])

# # sns.scatterplot(x=train[feature], y=train['1stFlrSF'], ax=axes[1,1])

# fig.tight_layout(pad=3.0)



# # (3)

# feature = 'SaleType'

# print('Null values =', train[feature].isnull().sum())

# print('highest % of any value =', int(train[feature].value_counts()[0]/len(train[feature])*100))

# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# sns.countplot(train[feature], ax=axes[0])

# sns.scatterplot(x=train[feature], y=train['SalePrice'], ax=axes[1])

# fig.tight_layout(pad=3.0)



# # (4)

# total_height = 4*len(num_features)

# fig, axes = plt.subplots(len(num_features), 3, figsize=(16, total_height))

# for i, nf in enumerate(num_features):

#     sns.distplot(train[nf].dropna(), ax=axes[i, 0], kde=False)

#     sns.boxplot(y=train[nf].dropna(), ax=axes[i, 1])

#     sns.regplot(x=train[nf], y=train['SalePrice'], ax=axes[i, 2])

# fig.tight_layout()