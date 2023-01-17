# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# save filepath to variable for easier access
Home_train_data_path = '/kaggle/input/home-data-for-ml-course/train.csv'
Home_test_data_path = '/kaggle/input/home-data-for-ml-course/test.csv'
# read the data and store data in DataFrame titled Home_data
train_data = pd.read_csv(Home_train_data_path,index_col='Id')
test_data = pd.read_csv(Home_test_data_path, index_col='Id')
print("Train shape : ", train_data.shape)
print("Test shape : ", test_data.shape)
# import the required libraries

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
pd.set_option('display.max_columns', 5000)
train_data.head()
pd.set_option('display.max_rows', 5000)
# columns with missing values
train_data[train_data.columns[train_data.isnull().any()]].isnull().sum()
# percentage of missing values
(train_data[train_data.columns[train_data.isnull().any()]].isnull().sum()* 100 / train_data.shape[0]).sort_values(axis=0, ascending=False)
# splitting the data into numerical and categorical.
numeric_col = train_data.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).copy()
cat_col = train_data.select_dtypes(include=['object']).copy()
cat_col['MSSubClass'] = train_data['MSSubClass']
#distribution of continuous numerical columns
disc_num_var = ['OverallQual','YearBuilt','YearRemodAdd','OverallCond','GarageYrBlt','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']
cont_num_var = []
for i in numeric_col.columns:
    if i not in disc_num_var:
        cont_num_var.append(i)

fig = plt.figure(figsize=(18,16))
for index,col in enumerate(cont_num_var):
    plt.subplot(6,4,index+1)
    sns.distplot(numeric_col.loc[:,col].dropna(), kde=False)
fig.tight_layout(pad=1.0)
# checking outliers for continuous numerical columns 
fig = plt.figure(figsize=(14,15))
for index,col in enumerate(cont_num_var):
    plt.subplot(6,4,index+1)
    sns.boxplot(y=col, data=numeric_col.dropna())
fig.tight_layout(pad=1.0)
#data distribution of discontinous numerical columns
fig = plt.figure(figsize=(20,15))
for index,col in enumerate(disc_num_var):
    plt.subplot(5,4,index+1)
    sns.countplot(x=col, data=numeric_col.dropna())
fig.tight_layout(pad=1.0)
#data distribution of categorical columns
fig = plt.figure(figsize=(18,20))
for index in range(len(cat_col.columns)):
    plt.subplot(9,5,index+1)
    sns.countplot(x=cat_col.iloc[:,index], data=cat_col.dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)
#correlation matrix
plt.figure(figsize=(20,15))
mask = numeric_col.corr() < 0.8

sns.heatmap(numeric_col.corr(),annot=True,mask = mask ,linewidth=0.7,fmt='.2g',cmap='Blues')
# Correlation with Target variable
numeric_col.corr()[['SalePrice']].sort_values(['SalePrice'],ascending = False)
# Removing outliers 
train_data = train_data.drop(train_data[train_data['LotFrontage'] > 200].index)
train_data = train_data.drop(train_data[train_data['LotArea'] > 100000].index)
train_data = train_data.drop(train_data[train_data['BsmtFinSF1'] > 4000].index)
train_data = train_data.drop(train_data[train_data['TotalBsmtSF'] > 5000].index)
train_data = train_data.drop(train_data[train_data['GrLivArea'] > 4000].index)
# combining train and test data
y = train_data['SalePrice'].reset_index(drop=True)
X_train = train_data.drop(['SalePrice'], axis=1)
X_test = test_data
X_all =  pd.concat([X_train, X_test]).reset_index(drop=True)
# Removing multicollinearity variables
X_all.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageArea'], axis=1, inplace=True)
# Removing variables with more than 90% missing values
X_all.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)
#Removing variables which are practically not available at the time of making predictions
X_all.drop(['MoSold','YrSold'], axis=1, inplace=True)
#practically less important
less_important = ['LotShape','LandSlope','LandContour']
X_all.drop(less_important, axis=1, inplace=True)
# Removing Categorical variables with mostly one value
cat_col = X_train.select_dtypes(include=['object']).columns
overfit_cat_col = []
for i in cat_col:
    counts = X_train[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X_train) * 100 > 96:
        overfit_cat_col.append(i)
overfit_cat_col = list(overfit_cat_col)
X_all.drop(overfit_cat_col, axis=1,inplace = True)
#Removing numerical variables with mostly one value
numeric_col = X_train.select_dtypes(exclude = ['object']).columns
overfit_num_col = []
for i in numeric_col:
    counts = X_train[i].value_counts()
    zeros = counts.iloc[0]
    if zeros/len(X_train)*100 > 95:
        overfit_num_col.append(i)
overfit_num_col = list(overfit_num_col)
X_all.drop(overfit_num_col,axis=1,inplace = True)
#changing data types of variables with incorrect datatypes
X_all['MSSubClass'] = X_all['MSSubClass'].apply(str)
#Mapping ordinal variables
ordinal_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu','BsmtFinType1','BsmtFinType2','Fence','BsmtExposure']
rating_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']
rating_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}

#Labeling ordinal variables
for col in rating_col:
    X_all[col] = X_all[col].map(rating_map)
    
fin_col = ['BsmtFinType1','BsmtFinType2']
for col in fin_col:
    X_all[col] = X_all[col].map(fintype_map) 

X_all['BsmtExposure'] = X_all['BsmtExposure'].map(expose_map)
X_all['Fence'] = X_all['Fence'].map(fence_map)
# Missing Numerical and categorical columns
num_col_with_missing_values = X_all[X_all.columns[X_all.isnull().any()]].select_dtypes(exclude=['object']).columns
Cat_col_with_missing_values = X_all[X_all.columns[X_all.isnull().any()]].select_dtypes(include=['object']).columns
# Making a copy of data for handling missing values
X_all_plus = X_all.copy()
X_all_plus = X_all.copy()
# Handling numerical missing values
#creating new columns to indicate the numerical missing values
for col in num_col_with_missing_values:
    X_all_plus[col + '_was_missing'] = X_all_plus[col].isnull()

#filling the numerical missing values with 0
X_all_plus[num_col_with_missing_values] = X_all_plus[num_col_with_missing_values].fillna(0)
#filling missing categorical values with "Missing"
X_all_plus[Cat_col_with_missing_values] = X_all_plus[Cat_col_with_missing_values].fillna("Missing")
#interaction features
X_all_plus['TotalLot'] = X_all_plus['LotFrontage'] + X_all_plus['LotArea']
X_all_plus['TotalBsmtFin'] = X_all_plus['BsmtFinSF1'] + X_all_plus['BsmtFinSF2']
X_all_plus['TotalBath'] = X_all_plus['FullBath'] + X_all_plus['HalfBath'] + X_all_plus['BsmtFullBath'] + X_all_plus['BsmtHalfBath']
X_all_plus['TotalPorch'] = X_all_plus['OpenPorchSF'] + X_all_plus['EnclosedPorch'] + X_all_plus['ScreenPorch']+ X_all_plus['WoodDeckSF']
#indicator features
X_all_plus['fireplace_available'] = X_all_plus['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#removing redundant features
redundant_features = ['LotFrontage','LotArea','BsmtFinSF1','Fireplaces','BsmtFinSF2','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath','ScreenPorch','EnclosedPorch','OpenPorchSF','WoodDeckSF']
X_all_plus.drop(redundant_features, axis=1, inplace=True)
#splitting train and test data
X = X_all_plus.iloc[:len(y), :]
test = X_all_plus.iloc[len(X):, :]
#creating training and validation datasets
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state = 42)
# Get list of categorical variables
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(valid_X[object_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(test[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = train_X.index
OH_cols_valid.index = valid_X.index
OH_cols_test.index = test.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = train_X.drop(object_cols, axis=1)
num_X_valid = valid_X.drop(object_cols, axis=1)
num_X_test = test.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

#final features data and variable
OH_X_final =  pd.concat([OH_X_train, OH_X_valid]).reset_index(drop=True)
y_final  = pd.concat([train_y,valid_y]).reset_index(drop=True)
def get_mae_xgb(X, y,learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,
            colsample_bytree=0.8,reg_alpha=0.005,scale_pos_weight=1):
    model = XGBRegressor(
 learning_rate =learning_rate,
 n_estimators=n_estimators,
 max_depth=max_depth,
 min_child_weight=min_child_weight,
 gamma=gamma,
 subsample=subsample,
 colsample_bytree=colsample_bytree,
 scale_pos_weight=scale_pos_weight,
reg_alpha=reg_alpha,
 objective= 'reg:squarederror',
    seed =42)
    scores = -1 * cross_val_score(model,X,y,cv=5,scoring='neg_mean_absolute_error')
    return scores.mean()
# fine tuning learning_rate
for learning_rate in [0.011,0.012,0.013]:
    my_mae = get_mae_xgb(X= OH_X_train, y=train_y,learning_rate =learning_rate)
    print("optimal learning_rate: %.4f  \t\t Mean Absolute Error:  %.4f" %(learning_rate, my_mae))
    
# fine tuning n_estimators
for n_estimators in range(1400,1800,100):
    my_mae = get_mae_xgb(X= OH_X_train, y=train_y,learning_rate =0.012,n_estimators=n_estimators)
    print("optimal n_estimators: %.4f  \t\t Mean Absolute Error:  %.4f" %(n_estimators, my_mae))

# fine tuning max_depth
for max_depth in range(4,7,1):
    my_mae = get_mae_xgb(X= OH_X_train, y=train_y,learning_rate =0.012,n_estimators=1600,max_depth=max_depth)
    print("optimal max_depth: %.4f  \t\t Mean Absolute Error:  %.4f" %(max_depth, my_mae))

# fine tuning min_child_weight
for min_child_weight in range(0,3,1):
    my_mae = get_mae_xgb(X= OH_X_train, y=train_y,learning_rate =0.012,n_estimators=1600,max_depth=4,min_child_weight=min_child_weight)
    print("optimal min_child_weight: %.4f  \t\t Mean Absolute Error:  %.4f" %(min_child_weight, my_mae))

# fine tuning gamma
for gamma in range(0,3,1):
    my_mae = get_mae_xgb(X= OH_X_train, y=train_y,learning_rate =0.012,n_estimators=1600,max_depth=4
                     ,min_child_weight=2,gamma=gamma)
    print("optimal gamma: %.4f  \t\t Mean Absolute Error:  %.4f" %(gamma, my_mae))

# fine tuning subsample
for subsample in [0.74,0.75,0.76]:
    my_mae = get_mae_xgb(X= OH_X_train, y=train_y,learning_rate =0.012,n_estimators=1600,max_depth=4
                     ,min_child_weight=2,gamma=0,subsample=subsample)
    print("optimal subsample: %.4f  \t\t Mean Absolute Error:  %.4f" %(subsample, my_mae))
    
# fine tuning colsample_bytree
for colsample_bytree in [0.77,0.78,0.79]:
    my_mae = get_mae_xgb(X= OH_X_train, y=train_y,learning_rate =0.012,n_estimators=1600,max_depth=4
                     ,min_child_weight=2,gamma=0,subsample=0.75,colsample_bytree=colsample_bytree)
    print("optimal colsample_bytree: %.4f  \t\t Mean Absolute Error:  %.4f" %(colsample_bytree, my_mae))
    
# fine tuning scale_pos_weight
for scale_pos_weight in [0,1,2]:
    my_mae = get_mae_xgb(X= OH_X_train, y=train_y,learning_rate =0.012,n_estimators=1600,max_depth=4
                     ,min_child_weight=2,gamma=0,subsample=0.75,colsample_bytree=0.79
                    ,scale_pos_weight=scale_pos_weight)
    print("optimal scale_pos_weight: %.4f  \t\t Mean Absolute Error:  %.4f" %(scale_pos_weight, my_mae))

# fine tuning reg_alpha
for reg_alpha in [0.005,0.05,0.5,0.1]:
    my_mae = get_mae_xgb(X= OH_X_train, y=train_y,learning_rate =0.012,n_estimators=1600,max_depth=4
                     ,min_child_weight=2,gamma=0,subsample=0.75,colsample_bytree=0.79
                    ,scale_pos_weight=1,reg_alpha=reg_alpha)
    print("optimal reg_alpha: %.4f  \t\t Mean Absolute Error:  %.4f" %(reg_alpha, my_mae))
# final xgboost model
xgboost = XGBRegressor(
    learning_rate=0.012,
    n_estimators=1600,
    max_depth=4,
    min_child_weight=2,
    gamma=0,
    subsample=0.75,
    colsample_bytree=0.79,
    scale_pos_weight=1,
    reg_alpha=0.005,
    seed=42
)
def get_mae_lgb(X, y,learning_rate=0.1,n_estimators=1000,feature_fraction=0.18,num_leaves=5,max_bin=180
            ,min_data_in_leaf=8,bagging_fraction = 0.35,bagging_freq =7):
    model = LGBMRegressor(objective='regression',
                          learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         feature_fraction_seed=42,
                         feature_fraction=feature_fraction,
                         num_leaves=num_leaves,
                          max_bin=max_bin,
                          min_data_in_leaf=min_data_in_leaf,
                         bagging_fraction=bagging_fraction,
                         bagging_seed=42,
                         bagging_freq=bagging_freq)
    scores = -1 * cross_val_score(model,X,y,cv=5,scoring='neg_mean_absolute_error')
    return scores.mean()
# fine tuning learning_rate
for learning_rate in [0.023,0.024,0.025]:
    my_mae = get_mae_lgb(X= OH_X_train, y=train_y,learning_rate =learning_rate)
    print("optimal learning_rate: %.4f  \t\t Mean Absolute Error:  %.4f" %(learning_rate, my_mae))

# fine tuning n_estimators
for n_estimators in range(2200,2800,100):
    my_mae = get_mae_lgb(X= OH_X_train, y=train_y,learning_rate =0.023,n_estimators=n_estimators)
    print("optimal n_estimators: %.4f  \t\t Mean Absolute Error:  %.8f" %(n_estimators, my_mae))
    
# fine tuning feature_fraction
for feature_fraction in [0.28,0.29,0.3,0.31,0.32]:
    my_mae = get_mae_lgb(X= OH_X_train, y=train_y,learning_rate =0.023,n_estimators=2400,
                     feature_fraction=feature_fraction)
    print("optimal feature_fraction: %.4f  \t\t Mean Absolute Error:  %.4f" %(feature_fraction, my_mae))
    
# fine tuning num_leaves
for num_leaves in range(3,8,1):
    my_mae = get_mae_lgb(X= OH_X_train, y=train_y,learning_rate =0.023,n_estimators=2400,
                     feature_fraction=0.3,num_leaves=num_leaves)
    print("optimal num_leaves: %.4f  \t\t Mean Absolute Error:  %.4f" %(num_leaves, my_mae))
    
# fine tuning max_bin
for max_bin in range(170,190,5):
    my_mae = get_mae_lgb(X= OH_X_train, y=train_y,learning_rate =0.023,n_estimators=2400,
                     feature_fraction=0.3,num_leaves=5,max_bin=max_bin)
    print("optimal max_bin: %.4f  \t\t Mean Absolute Error:  %.4f" %(max_bin, my_mae))
    
# fine tuning min_data_in_leaf
for min_data_in_leaf in range(5,10,1):
    my_mae = get_mae_lgb(X= OH_X_train, y=train_y,learning_rate =0.023,n_estimators=2400,
                     feature_fraction=0.3,num_leaves=5,max_bin=175,min_data_in_leaf=min_data_in_leaf)
    print("optimal min_data_in_leaf: %.4f  \t\t Mean Absolute Error:  %.4f" %(min_data_in_leaf, my_mae))
    
# fine tuning bagging_fraction
for bagging_fraction in [0.36,0.37,0.38]:
    my_mae = get_mae_lgb(X= OH_X_train, y=train_y,learning_rate =0.023,n_estimators=2400,
                     feature_fraction=0.3,num_leaves=5,max_bin=175,min_data_in_leaf=5
                    ,bagging_fraction=bagging_fraction)
    print("optimal bagging_fraction: %.4f  \t\t Mean Absolute Error:  %.4f" %(bagging_fraction, my_mae))
    
# fine tuning bagging_freq
for bagging_freq in range(5,10,1):
    my_mae = get_mae_lgb(X= OH_X_train, y=train_y,learning_rate =0.023,n_estimators=2400,
                     feature_fraction=0.3,num_leaves=5,max_bin=175,min_data_in_leaf=5
                    ,bagging_fraction=0.36,bagging_freq=bagging_freq)
    print("optimal bagging_freq: %.4f  \t\t Mean Absolute Error:  %.4f" %(bagging_freq, my_mae))
# final lightgbm model
lightgbm = LGBMRegressor(objective='regression',
                          learning_rate=0.023,
                         n_estimators=2400,
                         feature_fraction_seed=42,
                         feature_fraction=0.3,
                         num_leaves=5,
                          max_bin=175,
                          min_data_in_leaf=5,
                         bagging_fraction=0.36,
                         bagging_seed=42,
                         bagging_freq=7)
def get_mae_gbr(X, y,learning_rate=0.01,n_estimators=1000,max_depth=5,min_samples_leaf=15):
    model = GradientBoostingRegressor(learning_rate=learning_rate,
                                      n_estimators=n_estimators,
                                max_depth=max_depth,
                                max_features='sqrt',
                                min_samples_leaf=min_samples_leaf,
                                loss='huber',
                                random_state=42)
    scores = -1 * cross_val_score(model,X,y,cv=5,scoring='neg_mean_absolute_error')
    return scores.mean()
# fine tuning learning_rate
for learning_rate in [0.021,0.018]:
    my_mae = get_mae_gbr(X= OH_X_train, y=train_y,learning_rate =learning_rate)
    print("optimal learning_rate: %.4f  \t\t Mean Absolute Error:  %.4f" %(learning_rate, my_mae))
    
# fine tuning n_estimators
for n_estimators in range(1750,2500,250):
    my_mae = get_mae_gbr(X= OH_X_train, y=train_y,learning_rate =0.021,n_estimators=n_estimators)
    print("optimal n_estimators: %.4f  \t\t Mean Absolute Error:  %.4f" %(n_estimators, my_mae))
    
# fine tuning max_depth
for max_depth in range(4,8,1):
    my_mae = get_mae_gbr(X= OH_X_train, y=train_y,learning_rate =0.021,n_estimators=1750,max_depth=max_depth)
    print("optimal max_depth: %.4f  \t\t Mean Absolute Error:  %.4f" %(max_depth, my_mae))
    
# fine tuning min_samples_leaf
for min_samples_leaf in range(12,15,1):
    my_mae = get_mae_gbr(X= OH_X_train, y=train_y,learning_rate =0.021,n_estimators=1750,max_depth=4
                    ,min_samples_leaf=min_samples_leaf)
    print("optimal min_samples_leaf: %.4f  \t\t Mean Absolute Error:  %.4f" %(min_samples_leaf, my_mae))

# final gradientboosting model
gbr = GradientBoostingRegressor(learning_rate=0.021,
                                n_estimators=1750,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=13,
                                loss='huber',
                                random_state=42)

# stacking
stacking = StackingCVRegressor(regressors=(xgboost, lightgbm,gbr),
                                meta_regressor=gbr,
                                use_features_in_secondary=True)
scores = -1 * cross_val_score(stacking,OH_X_train.values,train_y.values,cv=5,scoring='neg_mean_absolute_error')
scores.mean()
# Fitting the models
xgb_model = xgboost.fit(OH_X_train, train_y)

lgb_model = lightgbm.fit(OH_X_train, train_y)

gbr_model = gbr.fit(OH_X_train, train_y)

stacking_model = stacking.fit(OH_X_train.values, train_y.values)
# blending the models
# find the optimum weights by trial and error
def blend_models_predict(X):
    return ((0.1 * gbr_model.predict(X)) +
            (0.2 * xgb_model.predict(X)) +
            (0.1 * lgb_model.predict(X)) +
            (0.6 * stacking_model.predict(X.values)))

mean_absolute_error(valid_y,blend_models_predict(OH_X_valid))
#final models
xgb_model_final = xgboost.fit(OH_X_final, y_final)

lgb_model_final = lightgbm.fit(OH_X_final, y_final)

gbr_model_final = gbr.fit(OH_X_final, y_final)

stacking_model_final = stacking.fit(OH_X_final.values, y_final.values)
#final weights for blending
def blend_models_predict(X):
    return ((0.35 * xgb_model_final.predict(X)) +
            (0.1 * lgb_model_final.predict(X)) +
            (0.1 * gbr_model_final.predict(X)) +
            (0.45 * stacking_model_final.predict(X.values)))
# Get predictions
predictions = blend_models_predict(OH_X_test)
Predictions = pd.DataFrame({'Id': test_data.index,
                      'SalePrice': predictions})
# Defining outlier quartile ranges
q1 = Predictions['SalePrice'].quantile(0.005)
q2 = Predictions['SalePrice'].quantile(0.99)

# Applying weights to outlier ranges to smooth them
Predictions['SalePrice'] = Predictions['SalePrice'].apply(
    lambda x: x if x > q1 else x * 0.77)
Predictions['SalePrice'] = Predictions['SalePrice'].apply(lambda x: x
                                                        if x < q2 else x * 1.1)
#submission
Predictions.to_csv('submission.csv', index=False)