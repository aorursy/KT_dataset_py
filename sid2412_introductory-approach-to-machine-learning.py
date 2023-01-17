import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from scipy.stats import norm, skew

from scipy import stats



pd.set_option('display.max_columns', 500)



import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

df_test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
train_id = df_train['Id']

test_id = df_test['Id']
df_train.drop('Id', axis=1, inplace=True)

df_test.drop('Id', axis=1, inplace=True)
print('Train dataset shape:', df_train.shape)

print('Test dataset shape:', df_test.shape)
# Creating num_cols to explore outliers in numerical columns 

num_cols = df_train.select_dtypes(exclude='object').drop('SalePrice', axis=1)
# Plotting scatter plots for every feature against Target

target = df_train['SalePrice']



f = plt.figure(figsize=(18,25))



for i in range(len(num_cols.columns)):

    f.add_subplot(9, 4, i+1)

    sns.scatterplot(num_cols.iloc[:,i], target)

    

plt.tight_layout()
# LotFrontage >200

df_train = df_train.drop(df_train[df_train['LotFrontage'] > 200].index)



# LotArea > 100000

df_train = df_train.drop(df_train[df_train['LotArea'] > 100000].index)



# BsmtFinSF1 > 3000

df_train = df_train.drop(df_train[df_train['BsmtFinSF1'] > 3000].index)



# BsmtFinSF2 > 1250

df_train = df_train.drop(df_train[df_train['BsmtFinSF2'] > 1250].index)



# TotalBsmtSF > 4000

df_train = df_train.drop(df_train[df_train['TotalBsmtSF'] > 4000].index)



# 1stFlrSF > 4000

df_train = df_train.drop(df_train[df_train['1stFlrSF'] > 4000].index)



# GrLivArea > 4000, SalePrice < 400000

df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 400000)].index)



# LowQualFinSF > 550

df_train = df_train.drop(df_train[df_train['LowQualFinSF'] > 550].index)
sns.distplot(target, fit=norm)

print('Skew is:', target.skew())
stats.probplot(target, plot=plt)
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit=norm)

print('Skew is:', df_train['SalePrice'].skew())
stats.probplot(df_train['SalePrice'], plot=plt)
ntrain = df_train.shape[0]

ntest = df_test.shape[0]

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.shape
all_data.isnull().sum().sort_values(ascending=False).head(40)
# PoolQC - filling with None as description saya NA for houses with no pools

all_data['PoolQC'] = all_data['PoolQC'].fillna('None')



# MiscFeature - filling with None as description saya NA for houses with no miscfeatures

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')



# Alley - filling with None as description saya NA for houses with no alley access

all_data['Alley'] = all_data['Alley'].fillna('None')



# Fence - filling with None as description saya NA for houses with no fence

all_data['Fence'] = all_data['Fence'].fillna('None')



# FireplaceQu - filling with None as description saya NA for houses with no fireplace

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')



# LotFrontage - filling in with median LotFrontage grouped by Neighborhood

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# Garage_cols - fillinf in with with None as description says NA for no garage

for col in ['GarageCond', 'GarageQual', 'GarageYrBlt', 'GarageFinish', 'GarageType']:

    all_data[col] = all_data[col].fillna('None')

all_data['GarageYrBlt'].replace('None', 0)



# Garage_cols - fillinf in with 0

for col in ['GarageArea', 'GarageCars']:

    all_data[col] = all_data[col].fillna(0)

    

# Bsmt_cols (categorical) - filling in with None

for col in ['BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']:

    all_data[col] = all_data[col].fillna('None')

    

# Bsmt_cols (numerical) - filling in with 0

for col in ['BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF2', 'BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF']:

    all_data[col] = all_data[col].fillna(0)



# MasVnrType - None, MasVnrArea - 0

all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)



# MSZoning - filling in with most occuring cat - RL

all_data['MSZoning'] = all_data['MSZoning'].fillna('RL')



# Functional = filling in with most occuring cat - Typ

all_data['Functional'] = all_data['Functional'].fillna('Typ')



# Exterior2nd - filling in with most occuring cat - VinylSd

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna('VinylSd')



# Exterior1st - filling in with most occuring cat - VinylSd

all_data['Exterior1st'] = all_data['Exterior1st'].fillna('VinylSd')



# SaleType - filling in with most occuring cat - WD

all_data['SaleType'] = all_data['SaleType'].fillna('WD')



# Electrical - filling in with most occuring cat - SBrkr

all_data['Electrical'] = all_data['Electrical'].fillna('SBrkr')



# KitchenQual - filling in with most occuring cat - TA

all_data['KitchenQual'] = all_data['KitchenQual'].fillna('TA')
# Utilities - drop the column 

all_data = all_data.drop(['Utilities'], axis=1)
# Checking for missing values

all_data.isnull().values.sum()
plt.figure(figsize=(15,10))

sns.heatmap(all_data.corr())
print(df_train.corr()['SalePrice'].sort_values(ascending=False))
all_data = all_data.drop(['GarageYrBlt', 'GarageCars', 'YearRemodAdd'], axis=1)
all_data.shape
# Object columns with high cardinality

high_card = [cname for cname in df_train.columns if (df_train[cname].nunique() > 10) & (df_train[cname].dtype == 'object')]
all_data[high_card].columns
from sklearn.preprocessing import LabelEncoder
# LabelEncoding the above columns



# Neighborhood

neigh_le = LabelEncoder()

all_data['Neighborhood'] = neigh_le.fit_transform(all_data['Neighborhood'])



# Exterior1st

ext1_le = LabelEncoder()

all_data['Exterior1st'] = ext1_le.fit_transform(all_data['Exterior1st'])



# Exterior2nd

ext2_le = LabelEncoder()

all_data['Exterior2nd'] = ext2_le.fit_transform(all_data['Exterior2nd'])
all_data.shape
# Object columns with low cardinality

low_card = [cname for cname in all_data.columns if  (all_data[cname].nunique() <= 10) & (all_data[cname].dtype == 'object')]
all_data[low_card].columns
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
num_feats = all_data.dtypes[all_data.dtypes != 'object'].index



skew_feats = all_data[num_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print('Skew of num_feats: ')

skewness = pd.DataFrame({'Skew' : skew_feats})

skewness.head()
skewness = skewness[abs(skewness) > 0.75]

print('Number of skewed features:', skewness.shape[0])
skewed_features = skewness.index

all_data[skewed_features] = np.log1p(all_data[skewed_features])
all_data.head()
all_data = pd.get_dummies(data=all_data, columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig',

       'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual', 'ExterCond',

       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

       'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',

       'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',

       'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',

       'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'], drop_first=True)
all_data.shape
X = all_data[:ntrain]

X_test = all_data[ntrain:]

y = df_train['SalePrice']
print('Train Null Values:', X.isnull().values.sum())

print('Test Null Values:', X_test.isnull().values.sum())

print('Train categorical features:', X.select_dtypes(include='object').values.sum())

print('Test categorical features:', X_test.select_dtypes(include='object').values.sum())
# Imports

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=100)
# Target is log(SalePrice). After prediction call, need to inverse transform to obtain SalePrice



def inv_y(trans_y):

    return np.exp(trans_y)
mae_comp = pd.Series()

mae_comp.index.name = 'Algorithm'
# Random Forest 



rf_model = RandomForestRegressor(random_state=100)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_val)

rf_mae = mean_absolute_error(inv_y(rf_preds), inv_y(y_val))

mae_comp['RandomForest'] = rf_mae
# Linear Regression



lin_model = LinearRegression()

lin_model.fit(X_train, y_train)

lin_preds = lin_model.predict(X_val)

lin_mae = mean_absolute_error(inv_y(lin_preds), inv_y(y_val))

mae_comp['LinearRegression'] = lin_mae
# Lasso



las_model = Lasso(alpha=0.00005, random_state=100)

las_model.fit(X_train, y_train)

las_preds = las_model.predict(X_val)

las_mae = mean_absolute_error(inv_y(las_preds), inv_y(y_val))

mae_comp['Lasso'] = las_mae
# KNNRegression



knn_model = KNeighborsRegressor()

knn_model.fit(X_train, y_train)

knn_preds = knn_model.predict(X_val)

knn_mae = mean_absolute_error(inv_y(knn_preds), inv_y(y_val))

mae_comp['KNN'] = knn_mae
# Ridge



ridge_model = Ridge(alpha=0.002, random_state=100)

ridge_model.fit(X_train, y_train)

ridge_preds = ridge_model.predict(X_val)

ridge_mae = mean_absolute_error(inv_y(ridge_preds), inv_y(y_val))

mae_comp['Ridge'] = ridge_mae
# ElasticNet



elas_model = ElasticNet(alpha=0.02, random_state=100, l1_ratio=0.7)

elas_model.fit(X_train, y_train)

elas_preds = elas_model.predict(X_val)

elas_mae = mean_absolute_error(inv_y(elas_preds), inv_y(y_val))

mae_comp['ElasticNet'] = elas_mae
# Gradient Boosting Regression



gbr_model = GradientBoostingRegressor(learning_rate=0.01, 

                                      subsample = 0.8, 

                                      n_estimators=1000, 

                                      max_depth=4

                                     )

gbr_model.fit(X_train, y_train)

gbr_preds = gbr_model.predict(X_val)

gbr_mae = mean_absolute_error(inv_y(gbr_preds), inv_y(y_val))

mae_comp['GradientBoosting'] = gbr_mae
mae_comp
mae_comp.min()
def score(model):

    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()

    return np.sqrt(-score)
scores= {}
scores.update({'RandomForest' : score(rf_model)})

scores.update({'LinearRegression' : score(lin_model)})

scores.update({'Lasso' : score(las_model)})

scores.update({'KNN' : score(knn_model)})

scores.update({'Ridge' : score(ridge_model)})

scores.update({'ElasticNet' : score(elas_model)})

scores.update({'GradientBoosting' : score(gbr_model)})
scores
min(scores.values())
final_model = GradientBoostingRegressor(learning_rate=0.01, 

                                      subsample = 0.8, 

                                      n_estimators=1000, 

                                      max_depth=4

                                     )

final_model.fit(X, y)

final_preds = final_model.predict(X_test)
output = pd.DataFrame({'Id': test_id,

                      'SalePrice' : inv_y(final_preds)})
output.to_csv('Submission.csv', index=False)