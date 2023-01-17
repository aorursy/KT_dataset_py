######################################################################################
# Handle warnings
######################################################################################
import warnings
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
warnings.simplefilter(action='ignore', category=FutureWarning) # Scipy warnings

######################################################################################
# Standard imports
######################################################################################
import os
import math
import pickle
from copy import copy
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from scipy import stats
from scipy.stats import norm
from scipy.stats import skew
from scipy.special import boxcox1p

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
train_path = "../input/train.csv"
test_path = "../input/test.csv"

# Load the dataset with Pandas
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Number of  samples
n_train = train.shape[0]
n_test = test.shape[0]

# Extract the GT labels and IDs from the data
train_y = train['SalePrice']
print(train_y.unique().shape)
train_ids = train['Id']
test_ids = test['Id']

train = train.drop('Id', axis=1)
test = test.drop('Id', axis=1)

# Split the dataset into continous and categorical features
train_numeric_feat = train.select_dtypes(include=[np.number]).columns.values

# Placeholder for columns that will be dropped at the end
drop_columns = []
fig = plt.figure(figsize=(15,3))
plt.subplot(121)
# Show distribution of labels against normal distribution
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.title('SalePrice')

plt.subplot(122)
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# Log transform the SalePrice and see the results
# Numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

fig = plt.figure(figsize=(15,3))
plt.subplot(121)
# Show distribution of labels against normal distribution
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.title('log(SalePrice)')

plt.subplot(122)
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
# Merge the test and train dataset to get access to all posibilities of categories
train_y = train['SalePrice'] 
train_wo_y = train.drop('SalePrice', axis=1)
all_data = pd.concat((train_wo_y, test)).reset_index(drop=True)
# Pool: 'NA' ==> No pool
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
# MiscFeatures: 'NA' ==> No additional features
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
# Alley: 'NA' ==> No alley access
all_data["Alley"] = all_data["Alley"].fillna("None")
# Fence: 'NA' ==> No fence
all_data["Fence"] = all_data["Fence"].fillna("None")
# FireplaceQu: 'NA' ==> No fireplace
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
# MSSubClass: 'NA' ==> 'NoBuilding'
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("NoBuildung")    

# LotFrontage: Calculate the median LotFrontage of the neighborhood
# The neighborhood attribute describes the area around the house
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# Garage attributes: 'NA' ==> Not given
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
# Additional attributes that are 'NA' if no garage is given ('NA')
# Replace these missing values with 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

# Basement: If there is no basement, these attributes are also 0 || Categorical features ==> None
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

# MasVnrType: If not given it will be None and nor area (0)
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# MSZoning: 'RL' is the most common value ==> fill empty fields with 'RL'
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# Functional: If 'NA' ==> typical
all_data["Functional"] = all_data["Functional"].fillna("Typical")

# Electrical: Most fields are 'SBrKr' ==> Fill empty fields with 'SBrKr'
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# Electrical: Most fields are 'TA' ==> Fill empty fields with 'TA'
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

# Exterior: Also apply most common categoreis
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# SaleType: Also apply most common categoreis
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# Utilities: All values are 'AllPub' except 3 different values ==> drop this attribute
drop_columns.append('Utilities')
perc_missing_col = (all_data.isnull().sum() / len(all_data)) * 100
perc_missing_col = perc_missing_col.drop(perc_missing_col[perc_missing_col == 0].index)
perc_missing_col = perc_missing_col.sort_values(ascending=False)[:20]
perc_missing_col = pd.DataFrame({'Missing Ratio' : perc_missing_col})
perc_missing_col.head() # The dataframe should be empty
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
def plot_df_distributions(df, num_cols=3, columns=None):
    if columns is None:
        _columns = df.loc[:,df.dtypes != 'object'].columns.values
    else:
        _columns = columns

    n_cols = num_cols
    n_rows = math.ceil(len(_columns)/n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*3.5,n_rows*3))

    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            col_idx = r_idx*3+c_idx
            
            if col_idx < len(_columns):
                col = _columns[col_idx]
                sns.distplot(df[col], ax=axes[r_idx][c_idx], fit=norm)
    
    plt.tight_layout()
    plt.show()
def plot_df_countplots(df, num_cols=3, columns=None):
    if columns is None:
        _columns = df.loc[:,df.dtypes == 'object'].columns.values
    else:
        _columns = columns

    n_cols = num_cols
    n_rows = math.ceil(len(_columns)/n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*4,n_rows*3))

    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            col_idx = r_idx*3+c_idx
            if col_idx < len(_columns):
                col = _columns[col_idx]
                
                sns.countplot(df[col], ax=axes[r_idx][c_idx])
                axes[r_idx][c_idx].set_title(col)
                for item in axes[r_idx][c_idx].get_xticklabels():
                    item.set_rotation(45)
    
    plt.tight_layout()
    plt.show()
plot_df_distributions(all_data, num_cols=3)
plot_df_countplots(all_data, num_cols=3)
# Mark some of the categorical features for dropping, because they are were unbalanced
drop_columns.append('Street')
drop_columns.append('Condition2')
drop_columns.append('RoofMatl')
drop_columns.append('Heating')
drop_columns.append('MiscVal')
TIME_FEATURES = [
    'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'
]

ORDERD_CAT_FEATURES = [
    'LotShape', 'LandSlope', 'OverallQual', 'PoolQC', 'GarageQual',
    'Fence', 'GarageCond', 'GarageFinish', 'OverallCond', 'ExterQual', 'ExterCond',
    'KitchenQual', 'FireplaceQu', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'BsmtExposure'
]

SIMPLE_CAT_FEATURES = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'SaleType',
    'SaleCondition', 'PavedDrive', 'MiscFeature', 'MiscVal', 'LotConfig',
    'Neighborhood', 'Electrical', 'Functional', 'GarageType', 'CentralAir',
    'Heating', 'Foundation', 'MasVnrType', 'Exterior2nd', 'Exterior1st',
    'RoofMatl', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
    'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'MoSold', 'Utilities'
]

ALL_CAT_FEATURES = TIME_FEATURES + ORDERD_CAT_FEATURES + SIMPLE_CAT_FEATURES

CONTINUOUS_FEATURES = list(all_data[all_data.columns.difference(ALL_CAT_FEATURES)].columns.values)

print('Total number of columns: {}'.format(len(CONTINUOUS_FEATURES) + len(SIMPLE_CAT_FEATURES) + len(ORDERD_CAT_FEATURES) + len(TIME_FEATURES)))
for col in TIME_FEATURES:
    all_data[col] = all_data[col].astype(int)
all_data[TIME_FEATURES].head(5)
ORDERED_ENCODINGS = {
    'LotShape' : {'Reg' : 3, 'IR1' : 2, 'IR2': 1, 'IR3': 0},
    'LandSlope' : {'Gtl' : 2, 'Mod' : 1, 'Sev' : 0},
    'OverallQual' : {'10' : 9, '9' : 8, '8' : 7, '7' : 6, '6' : 5, '5' : 4, '4' : 3, '3' : 2, '2' : 1, '1' : 0},
    'PoolQC' : {'None' : 0, 'Fa' : 1, 'TA' : 2, 'Gd' : 3, 'Ex' : 4},
    'GarageQual' : {'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'Fence' : {'None' : 0, 'MnWw' : 1, 'GdWo' : 2, 'MnPrv' : 3, 'GdPrv' : 4},
    'GarageCond' : {'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'GarageFinish' : {'None' : 0, 'Unf' : 1, 'RFn' : 2, 'Fin' : 3},
    'OverallCond' : {'10' : 9, '9' : 8, '8' : 7, '7' : 6, '6' : 5, '5' : 4, '4' : 3, '3' : 2, '2' : 1, '1' : 0},
    'ExterQual' : {'None' : 0, 'Fa' : 1, 'TA' : 2, 'Gd' : 3, 'Ex' : 4}, 
    'ExterCond' : {'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'KitchenQual' : {'None' : 0, 'Fa' : 1, 'TA' : 2, 'Gd' : 3, 'Ex' : 4},
    'FireplaceQu' : {'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'BsmtQual' : {'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'BsmtCond' : {'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'HeatingQC' : {'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'BsmtExposure' : {'None' : 0, 'No' : 1, 'Mn' : 2, 'Av' : 3, 'Gd' : 4},
    'BsmtFinType1' : {'None' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6},
    'BsmtFinType2' : {'None' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6},
    #'MoSold' : {'1' : 0, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '11' : 11, '12' : 12}
}

for col in ORDERD_CAT_FEATURES:
    all_data[col] = all_data[col].apply(str)
    all_data[col] = all_data[col].map(ORDERED_ENCODINGS[col])
    all_data[col] = all_data[col].dropna(0)
    all_data[col] = all_data[col].apply(float)
    
all_data[ORDERD_CAT_FEATURES].head(5)
for col in SIMPLE_CAT_FEATURES:
    all_data[col] = all_data[col].astype(str)
all_data[SIMPLE_CAT_FEATURES].head(5)
feature_skewness = all_data[CONTINUOUS_FEATURES].apply(lambda x: skew(x.dropna()))
feature_skewness = feature_skewness.sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' : feature_skewness})
skewness.head(10)
high_skewed_features = skewness.loc[abs(skewness.Skew) >= 1.0].index.values
plot_df_distributions(all_data, columns=high_skewed_features)
# Fix skew with box_cox transfomation
lam = 0.15
for feat in high_skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
plot_df_distributions(all_data, columns=CONTINUOUS_FEATURES)
# Scale the data
features_to_scale = CONTINUOUS_FEATURES + TIME_FEATURES + ORDERD_CAT_FEATURES
numeric_scaler = MinMaxScaler()
numeric_scaler.fit(all_data[features_to_scale])
all_data[features_to_scale] = numeric_scaler.transform(all_data[features_to_scale])
plot_df_distributions(all_data)
all_data = all_data.drop(drop_columns, axis=1)
all_data_oh = pd.get_dummies(all_data)
train = all_data_oh[:n_train]
test = all_data_oh[n_train:]
print("Shape train: {}".format(train.shape))
print("Shape test: {}".format(test.shape))
train.head(5)
test.head(5)
isolation_forest = IsolationForest(max_samples=100, random_state=42)
isolation_forest.fit(train)

# Evaluate all rows by the random forest
outlier_info = pd.DataFrame(isolation_forest.predict(train), columns=['Top'])

# Get the index of the outliers and remove them 
no_outlier_idxs = outlier_info[outlier_info['Top'] == 1].index.values
outlier_idxs = outlier_info[outlier_info['Top'] == -1].index.values
train = train.iloc[no_outlier_idxs]

# Also remove the outliers from the labels (Y)
train_y = train_y.iloc[no_outlier_idxs]

print('Number of outliers: {}'.format(outlier_idxs.shape[0]))
print('Shape train dataset after removal: {}'.format(train.shape[0]))
print('Shape train dataset labes after removal: {}'.format(train_y.shape[0]))
dataset_container = {
    'train' : train,
    'train_y' : train_y,
    'train_ids' : train_ids,
    'test' : test,
    'test_ids' : test_ids,
    
    # Column names by type
    'time_based_features' : TIME_FEATURES,
    'ordered_category_features' : ORDERD_CAT_FEATURES,
    'simple_category_features' : SIMPLE_CAT_FEATURES,
    'numeric_features' : CONTINUOUS_FEATURES,
    
    # Encoding dicts
    'ordered_category_features_encodings' : ORDERED_ENCODINGS,
    'scaler' : numeric_scaler
}


train['SalePrice'] = train_y
train['Id'] = train_ids
train.to_csv('preprocessed_train.csv', index=False)
test['Id'] = test_ids.values
test.to_csv('preprocessed_test.csv', index=False)

# Dump the results
#with open('res/basic_house_pricing/preprocessed_data-ADVANCED-ENGINEERING.cdump', 'wb') as out_file:
#    pickle.dump(obj=dataset_container, file=out_file)