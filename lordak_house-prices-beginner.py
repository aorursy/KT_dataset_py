import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np

import pandas as pd



from datetime import datetime

from scipy.stats import probplot

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



import warnings

warnings.filterwarnings('ignore')



SEED = 42

PATH = '/kaggle/input/house-prices-advanced-regression-techniques/'
# Returns a concatenated df of training and test set on axis 0

def concat_df(train_data, test_data):    

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



# Returns divided dfs of training and test set

def divide_df(all_data):    

    return all_data.loc[:1459], all_data.loc[1460:].drop(['SalePrice'], axis=1)



df_train = pd.read_csv(PATH + 'train.csv')

df_test = pd.read_csv(PATH + 'test.csv')

df_all = concat_df(df_train, df_test)

dfs = [df_train, df_test]



# Filling masonry veneer features

df_all['MasVnrArea'] = df_all['MasVnrArea'].fillna(0)

df_all['MasVnrType'] = df_all['MasVnrType'].fillna('None')



# Filling continuous basement features

for feature in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:

    df_all[feature] = df_all[feature].fillna(0)



# Filling categorical basement features

for feature in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:

    df_all[feature] = df_all[feature].fillna('None')



# Filling continuous garage features

for feature in ['GarageArea', 'GarageCars', 'GarageYrBlt']:

    df_all[feature] = df_all[feature].fillna(0)



# Filling categorical garage features

for feature in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    df_all[feature] = df_all[feature].fillna('None')

    

# Filling other categorical features

for feature in ['Alley', 'Fence', 'FireplaceQu', 'MiscFeature', 'PoolQC']:

    df_all[feature] = df_all[feature].fillna('None')



# Filling missing values in categorical features with the mode value of neighborhood and house type

for feature in ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual', 'MSZoning', 'SaleType', 'Utilities']:

    df_all[feature] = df_all.groupby(['Neighborhood', 'MSSubClass'])[feature].apply(lambda x: x.fillna(x.mode()[0]))



# Filling the missing values in LotFrontage with the median of neighborhood

df_all['LotFrontage'] = df_all.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))
# Creating custom features



df_all['YearBuiltRemod'] = df_all['YearBuilt'] + df_all['YearRemodAdd']

df_all['TotalSF'] = df_all['TotalBsmtSF'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']

df_all['TotalSquareFootage'] = df_all['BsmtFinSF1'] + df_all['BsmtFinSF2'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']

df_all['TotalBath'] = df_all['FullBath'] + (0.5 * df_all['HalfBath']) + df_all['BsmtFullBath'] + (0.5 * df_all['BsmtHalfBath'])

df_all['TotalPorchSF'] = df_all['OpenPorchSF'] + df_all['3SsnPorch'] + df_all['EnclosedPorch'] + df_all['ScreenPorch'] + df_all['WoodDeckSF']

df_all['OverallRating'] = df_all['OverallQual'] + df_all['OverallCond']



df_all['HasPool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df_all['Has2ndFloor'] = df_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df_all['HasGarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df_all['HasBsmt'] = df_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df_all['HasFireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



df_all['NewHouse'] = 0

idx = df_all[df_all['YrSold'] == df_all['YearBuilt']].index

df_all.loc[idx, 'NewHouse'] = 1
# Outlier’s detection and deletion



df_all.drop(df_all[np.logical_and(df_all['OverallQual'] < 5, df_all['SalePrice'] > 200000)].index, inplace=True)

df_all.drop(df_all[np.logical_and(df_all['GrLivArea'] > 4000, df_all['SalePrice'] < 300000)].index, inplace=True)

df_all.reset_index(drop=True, inplace=True)
# Encoding ordinal features



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



df_all['BsmtCond'] = df_all['BsmtCond'].map(bsmtcond_map)

df_all['BsmtExposure'] = df_all['BsmtExposure'].map(bsmtexposure_map)

df_all['BsmtFinType1'] = df_all['BsmtFinType1'].map(bsmtfintype_map)

df_all['BsmtFinType2'] = df_all['BsmtFinType2'].map(bsmtfintype_map)

df_all['BsmtQual'] = df_all['BsmtQual'].map(bsmtqual_map)

df_all['CentralAir'] = df_all['CentralAir'].map(centralair_map)

df_all['ExterCond'] = df_all['ExterCond'].map(extercond_map)

df_all['ExterQual'] = df_all['ExterQual'].map(exterqual_map)

df_all['FireplaceQu'] = df_all['FireplaceQu'].map(fireplacequ_map)

df_all['Functional'] = df_all['Functional'].map(functional_map)

df_all['GarageCond'] = df_all['GarageCond'].map(garagecond_map)

df_all['GarageFinish'] = df_all['GarageFinish'].map(garagefinish_map)

df_all['GarageQual'] = df_all['GarageQual'].map(garagequal_map)

df_all['HeatingQC'] = df_all['HeatingQC'].map(heatingqc_map)

df_all['KitchenQual'] = df_all['KitchenQual'].map(kitchenqual_map)

df_all['LandSlope'] = df_all['LandSlope'].map(landslope_map)

df_all['LotShape'] = df_all['LotShape'].map(lotshape_map)

df_all['PavedDrive'] = df_all['PavedDrive'].map(paveddrive_map)
nominal_features = ['Alley', 'BldgType', 'Condition1', 'Condition2', 'Electrical', 

                    'Exterior1st', 'Exterior2nd', 'Fence', 'Foundation', 'GarageType', 

                    'Heating', 'HouseStyle', 'LandContour', 'LotConfig', 'MSSubClass',

                    'MSZoning', 'MasVnrType', 'MiscFeature', 'MoSold', 'Neighborhood',

                    'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'YrSold']
# One-hot encoding for nominal or nominal+ordinal features



encoded_features = []



for feature in nominal_features:

    encoded_df = pd.get_dummies(df_all[feature])

    n = df_all[feature].nunique()

    encoded_df.columns = ['{}_{}'.format(feature, col) for col in encoded_df.columns]

    encoded_features.append(encoded_df)



df_all = pd.concat([df_all, *encoded_features], axis=1)

df_all.drop(columns=nominal_features, inplace=True)



df_all.drop(columns=['Street', 'Utilities', 'PoolQC'], inplace=True)
# Transforming highly skewed features



cont_features = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2',

                 'BsmtUnfSF', 'EnclosedPorch', 'GarageArea', 'GrLivArea', 'LotArea', 

                 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 

                 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF']



skewed_features = {feature: df_all[feature].skew() for feature in cont_features if df_all[feature].skew() >= 0.5}

transformed_skews = {}



for feature in skewed_features.keys():

    df_all[feature] = boxcox1p(df_all[feature], boxcox_normmax(df_all[feature] + 1))

    transformed_skews[feature] = df_all[feature].skew()

    

df_skew = pd.DataFrame(index=skewed_features.keys(), columns=['Skew', 'Skew after boxcox1p'])

df_skew['Skew'] = skewed_features.values()

df_skew['Skew after boxcox1p'] = transformed_skews.values()
# Running LassoCV



df_all['SalePrice'] = np.log1p(df_all['SalePrice'])



df_train, df_test = df_all.loc[:1456], df_all.loc[1457:].drop(['SalePrice'], axis=1)

drop_cols = ['Id']

X_train = df_train.drop(columns=drop_cols + ['SalePrice']).values

y_train = df_train['SalePrice'].values

X_test = df_test.drop(columns=drop_cols).values



def cv_rmse(model, X=X_train, y=y_train):

    return np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf))



K = 10

kf = KFold(n_splits=K, shuffle=True, random_state=SEED)



model = make_pipeline(RobustScaler(), LassoCV(alphas=np.arange(0.0001, 0.0009, 0.0001), random_state=SEED, cv=kf))

model.fit(X_train, y_train)

predictions = np.expm1(model.predict(X_train))

    

score = cv_rmse(model, X=X_train, y=y_train)

scores = (score.mean(), score.std())

    

print(' Finished Running LassoCV')

print(' Mean RMSE: {:.6f} / Std: {:.6f}\n'.format(scores[0], scores[1]))
# Save results



submission_df = pd.DataFrame(columns=['Id', 'SalePrice'])

submission_df['Id'] = df_test['Id']

submission_df['SalePrice'] = np.expm1(model.predict(X_test))

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)