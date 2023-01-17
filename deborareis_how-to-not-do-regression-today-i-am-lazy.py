# Import

import pandas as pd

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()
features = train

features['Functional'] = features['Functional'].fillna('Typ')

features['Electrical'] = features['Electrical'].fillna("SBrkr")

features['KitchenQual'] = features['KitchenQual'].fillna("TA")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

features["PoolQC"] = features["PoolQC"].fillna("None")



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')



features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)



features.update(features[objects].fillna('None'))

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# Filling in the rest of the NA's

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))



features = features.drop(['Utilities', 'Street', 'PoolQC'], axis=1)

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



# simplified features

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



final_features = features

print(final_features.shape)
# Apply label encoder do all object features

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()



object_dtypes = ['object']

objects_labelencoded = []

for i in final_features.columns:

    if final_features[i].dtype in object_dtypes:

        objects_labelencoded.append(i)

        

for i in objects_labelencoded:

    final_features[i] = lb_make.fit_transform(final_features[i])

final_features.shape
# Identify skewed features

from scipy.stats import norm, skew

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in final_features.columns:

    if final_features[i].dtype in numeric_dtypes: numerics2.append(i)

        

skew_features = final_features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

print(skew_features)



# BoxCox

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index

for i in skew_index: final_features[i] = boxcox1p(final_features[i], boxcox_normmax(final_features[i] + 1))

final_features.shape
y = final_features.SalePrice

X = final_features.drop(columns=['SalePrice'])



from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import numpy as np 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)



from sklearn.ensemble import GradientBoostingRegressor

est = GradientBoostingRegressor() 

est.fit(X_train, y_train)

y_pred = est.predict(X_test)



print("Mean Sale Price: ", np.mean(train.SalePrice))

print("R2: ", r2_score(y_test, y_pred))  

print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred))) 
# Feature Importance

feat_importances = pd.Series(est.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh');
# TEST

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



features = test

features['Functional'] = features['Functional'].fillna('Typ')

features['Electrical'] = features['Electrical'].fillna("SBrkr")

features['KitchenQual'] = features['KitchenQual'].fillna("TA")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

features["PoolQC"] = features["PoolQC"].fillna("None")



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')



features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)



features.update(features[objects].fillna('None'))

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# Filling in the rest of the NA's

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))



features = features.drop(['Utilities', 'Street', 'PoolQC'], axis=1)

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



# simplified features

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



final_features = features



# Label Encode

for i in objects_labelencoded:

    final_features[i] = lb_make.fit_transform(final_features[i])



# BoxCox

list_skew = list(skew_index)

list_skew.remove('SalePrice')

for i in list_skew: final_features[i] = boxcox1p(final_features[i], boxcox_normmax(final_features[i] + 1))

final_features.shape
# Predict on Test

final_features['SalePrice'] = est.predict(final_features)

final_features.head()
# Save results to submit

test = final_features[['Id','SalePrice']]

test.head()

test.to_csv("submission_regression.csv", index=False)