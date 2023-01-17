import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import probplot

from sklearn.model_selection import KFold, cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LassoCV

from xgboost import XGBRegressor
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



print('Number of Training Examples = {}'.format(train.shape[0]))

print('Number of Test Examples = {}\n'.format(test.shape[0]))

print('Training X Shape = {}'.format(train.shape))

print('Training y Shape = {}\n'.format(train['SalePrice'].shape[0]))

print('Test X Shape = {}'.format(test.shape))

print('Test y Shape = {}\n'.format(test.shape[0]))
def get_data_for_analysis():

    return pd.concat([train, test], sort=True)
get_data_for_analysis().info()
#descriptive statistics summary

train['SalePrice'].describe()
data = get_data_for_analysis()

for column in data.columns.tolist():

    if data[column].isnull().sum():

        print("{} column has {} missing value from {}".format(column, data[column].isnull().sum(), len(data)))
get_data_for_analysis()['MasVnrType']
def filling_mansory_features(data):    

    # Filling masonry veneer features

    data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

    data['MasVnrType'] = data['MasVnrType'].fillna('None')



    # Filling continuous basement features

    for feature in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:

        data[feature] = data[feature].fillna(0)



    # Filling categorical basement features

    for feature in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:

        data[feature] = data[feature].fillna('None')



    # Filling continuous garage features

    for feature in ['GarageArea', 'GarageCars', 'GarageYrBlt']:

        data[feature] = data[feature].fillna(0)



    # Filling categorical garage features

    for feature in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

        data[feature] = data[feature].fillna('None')



    # Filling other categorical features

    for feature in ['Alley', 'Fence', 'FireplaceQu', 'MiscFeature', 'PoolQC']:

        data[feature] = data[feature].fillna('None')

filling_mansory_features(train)

filling_mansory_features(test)



data =  pd.concat([train, test], sort=True)

for column in data.columns.tolist():

    if data[column].isnull().sum():

        print("{} column has {} missing value from {}".format(column, data[column].isnull().sum(), len(data)))
data.head()
train.drop(columns=['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual', 'MSZoning', 'SaleType', 'Utilities', 'LotFrontage'], inplace=True)

test.drop(columns=['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual', 'MSZoning', 'SaleType', 'Utilities', 'LotFrontage'], inplace=True)
data =  pd.concat([train, test], sort=True)

for column in data.columns.tolist():

    if data[column].isnull().sum():

        print("{} column has {} missing value from {}".format(column, data[column].isnull().sum(), len(data)))
train.drop(train[np.logical_and(train['OverallQual'] < 5, train['SalePrice'] > 200000)].index, inplace=True)

train.drop(train[np.logical_and(train['GrLivArea'] > 4000, train['SalePrice'] < 300000)].index, inplace=True)

train.drop(columns=['Street', 'PoolQC'], inplace=True)

test.drop(columns=['Street', 'PoolQC'], inplace=True)
train.info()
bsmtcond_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4}

bsmtexposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

bsmtfintype_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

bsmtqual_map = {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}

centralair_map = {'Y': 1, 'N': 0}

extercond_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

exterqual_map = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}

fireplacequ_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

functional_map = {'Typ': 0, 'Min1': 1, 'Min2': 1, 'Mod': 2, 'Maj1': 3, 'Maj2': 3, 'Sev': 4}

garagecond_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

garagefinish_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}

garagequal_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

heatingqc_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

kitchenqual_map = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}

landslope_map = {'Gtl': 1, 'Mod': 2, 'Sev': 3}

lotshape_map = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}

paveddrive_map = {'N': 0, 'P': 1, 'Y': 2}

def perform_map(df_all):

    df_all['BsmtCond'] = df_all['BsmtCond'].map(bsmtcond_map)

    df_all['BsmtExposure'] = df_all['BsmtExposure'].map(bsmtexposure_map)

    df_all['BsmtFinType1'] = df_all['BsmtFinType1'].map(bsmtfintype_map)

    df_all['BsmtFinType2'] = df_all['BsmtFinType2'].map(bsmtfintype_map)

    df_all['BsmtQual'] = df_all['BsmtQual'].map(bsmtqual_map)

    df_all['CentralAir'] = df_all['CentralAir'].map(centralair_map)

    df_all['ExterCond'] = df_all['ExterCond'].map(extercond_map)

    df_all['ExterQual'] = df_all['ExterQual'].map(exterqual_map)

    df_all['FireplaceQu'] = df_all['FireplaceQu'].map(fireplacequ_map)

    df_all['GarageCond'] = df_all['GarageCond'].map(garagecond_map)

    df_all['GarageFinish'] = df_all['GarageFinish'].map(garagefinish_map)

    df_all['GarageQual'] = df_all['GarageQual'].map(garagequal_map)

    df_all['HeatingQC'] = df_all['HeatingQC'].map(heatingqc_map)

    df_all['LandSlope'] = df_all['LandSlope'].map(landslope_map)

    df_all['LotShape'] = df_all['LotShape'].map(lotshape_map)

    df_all['PavedDrive'] = df_all['PavedDrive'].map(paveddrive_map)

    

perform_map(train)

perform_map(test)
nominal_features = ['Alley', 'BldgType', 'Condition1', 'Condition2', 'Fence', 'Foundation', 'GarageType', 

                    'Heating', 'HouseStyle', 'LandContour', 'LotConfig', 'MSSubClass',

                    'MasVnrType', 'MiscFeature', 'MoSold', 'Neighborhood',

                    'RoofMatl', 'RoofStyle', 'SaleCondition', 'YrSold']



train.drop(columns=nominal_features, inplace=True)

test.drop(columns=nominal_features, inplace=True)
train['SalePrice'] = np.log1p(train['SalePrice'])
drop_cols = ['Id']

X_train = train.drop(columns=drop_cols + ['SalePrice']).values

y_train = train['SalePrice'].values

X_test = test.drop(columns=drop_cols).values



print('X_train shape: {}'.format(X_train.shape))

print('y_train shape: {}'.format(y_train.shape))

print('X_test shape: {}'.format(X_test.shape))
def rmse(y_train, y_pred):

     return np.sqrt(mean_squared_error(y_train, y_pred))



def cv_rmse(model, X=X_train, y=y_train):    

    return np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf))



K = 10

kf = KFold(n_splits=K, shuffle=True, random_state=42)
train
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=np.arange(0.0001, 0.0009, 0.0001), random_state=42, cv=kf))

models = {'lasso': lasso}

predictions = {}

scores = {}



for name, model in models.items():

    print('Running {}'.format(name))

    

    model.fit(X_train, y_train)

    predictions[name] = np.expm1(model.predict(X_train))

    

    score = cv_rmse(model, X=X_train, y=y_train)

    scores[name] = (score.mean(), score.std())

    

    print(' Finished Running {}'.format(name))

    print(' {} Mean RMSE: {:.6f} / Std: {:.6f}\n'.format(name, scores[name][0], scores[name][1]))
submission_df = pd.DataFrame(columns=['Id', 'SalePrice'])

submission_df['Id'] = test['Id']

print(test['Id'].shape)

print(X_test.shape)

submission_df['SalePrice'] = np.expm1(lasso.predict(X_test))

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)

print(submission_df.shape)