# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# Plotting Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries With scikit Learn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

#for Statistics
import scipy
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
# Reading The dataset
hp_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
hp_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
hp_train.shape, hp_test.shape
hp_train.head()
hp_test.head()
# Finding Target Column
hp_train.columns.difference(hp_test.columns)
# Checking The Distribution of Train DataSet
sns.set_style("dark")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(10, 8))
sns.distplot(hp_train['SalePrice'], color="green");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="Housing SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()
# exploring datatype and unique values for each column/feature
list(zip(hp_train.columns,hp_train.dtypes,hp_train.nunique(dropna=False)))
# Finding numeric features
numeric_dtypes = ['int64','float64']
numeric = []
for i in hp_train.columns:
    if hp_train[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms','Total_porch_sf','haspool','hasgarage','hasbsmt','hasfireplace']:
            pass
        else:
            numeric.append(i)     
# visualising some more outliers in the data values
fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 120))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(hp_train[numeric]), 1):
    if(feature=='MiscVal'):
        break
    plt.subplot(len(list(numeric)), 3, i)
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=hp_train)
        
    plt.xlabel('{}'.format(feature), size=20,labelpad=12.5)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 10})
        
plt.show()
corr = hp_train.corr()
plt.subplots(figsize=(20,16))
sns.heatmap(corr, vmax=0.8, cmap="magma", square=True)
# OverallQual to the SalePrice
data = pd.concat([hp_train['SalePrice'], hp_train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.barplot( x=hp_train['OverallQual'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
# YearBuilt to the SalePrice
data = pd.concat([hp_train['SalePrice'], hp_train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(20, 10))
fig = sns.barplot( x=hp_train['YearBuilt'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
# Ground Living area(GrLivArea) To SalePrice

data = pd.concat([hp_train['SalePrice'], hp_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', alpha=0.5, ylim=(0,800000), figsize=(20, 10));
# Basement Surface Area(TotalBsmtSF) To SalePrice

data = pd.concat([hp_train['SalePrice'], hp_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', alpha=0.5, ylim=(0,800000), figsize=(20, 10));
#LotArea To SalePrice

data = pd.concat([hp_train['SalePrice'], hp_train['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice', alpha=0.5, ylim=(0,800000), figsize=(20, 10));

# Remove the Ids from train and test, as they are unique for each row and hence not useful for the model

train_ID = hp_train['Id']
test_ID = hp_test['Id']
hp_train.drop(['Id'], axis=1, inplace=True)
hp_test.drop(['Id'], axis=1, inplace=True)
hp_train.shape, hp_test.shape
# Checking The Distribution of Train DataSet
sns.set_style("dark")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(12, 9))
sns.distplot(hp_train['SalePrice'], color="green");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="Housing SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()

#log(1+x) transform
hp_train["SalePrice"] = np.log1p(hp_train["SalePrice"])
sns.set_style("dark")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(12, 9))
#Check the new distribution 
sns.distplot(hp_train['SalePrice'] , fit=norm, color="green");

# Get the fitted parameters used by the function
(mean, sigma) = norm.fit(hp_train['SalePrice'])
print( '\n mean = {:.2f} and sigma = {:.2f}\n'.format(mean, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mean, sigma)],loc='best')
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)

plt.show()
# Remove the Outluiers.
hp_train.drop(hp_train[(hp_train['OverallQual']<5) & (hp_train['SalePrice']>200000)].index, inplace=True)
hp_train.drop(hp_train[(hp_train['GrLivArea']>4500) & (hp_train['SalePrice']<300000)].index, inplace=True)
hp_train.reset_index(drop=True, inplace=True)
hp_train.shape
# Split features and labels
hp_train_labels = hp_train['SalePrice'].reset_index(drop=True)
hp_train_features = hp_train.drop(['SalePrice'], axis=1)
hp_test_features = hp_test

# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
hp = pd.concat([hp_train_features, hp_test_features]).reset_index(drop=True)
hp.shape
cat_cols = hp.select_dtypes(['object']).columns
print("Categorical columns: ",cat_cols.tolist())
(hp[cat_cols].isnull().sum())*100/hp.shape[0]
# determine the threshold for missing values
def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return dict_x

missing = percent_missing(hp)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]
# Visualize missing values
sns.set_style("dark")
f, ax = plt.subplots(figsize=(12, 9))
sns.set_color_codes(palette='deep')
missing = round(hp_train.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="purple")
# Tweak the visual presentation
ax.xaxis.grid(False)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)
hp['Alley'] = np.where(hp['Alley'].isnull(),"No_Alley_Access",hp['Alley'])

hp['FireplaceQu'] = np.where(hp['FireplaceQu'].isnull(),"No_Fireplace",hp['FireplaceQu'])

hp['PoolQC'] = np.where(hp['PoolQC'].isnull(),"No_Pool",hp['PoolQC'])

hp['Fence'] = np.where(hp['Fence'].isnull(),"No_Fence",hp['Fence'])

hp['MiscFeature'] = np.where(hp['MiscFeature'].isnull(),"No_MiscFeature",hp['MiscFeature'])

hp['MSZoning'] = np.where(hp['MSZoning'].isnull(),"RL",hp['MSZoning'])

hp['Exterior1st'] = np.where(hp['Exterior1st'].isnull(),"VinylSd",hp['Exterior1st'])

hp['Exterior2nd'] = np.where(hp['Exterior2nd'].isnull(),"VinylSd",hp['Exterior2nd'])

hp['MasVnrType'] = np.where(hp['MasVnrType'].isnull(),"None",hp['MasVnrType'])

hp['BsmtQual'] = np.where(hp['BsmtQual'].isnull(),"None",hp['BsmtQual'])

hp['BsmtCond'] = np.where(hp['BsmtCond'].isnull(),"NoBasement",hp['BsmtCond'])

hp['BsmtExposure'] = np.where(hp['BsmtExposure'].isnull(),"NoBasement",hp['BsmtExposure'])

hp['BsmtFinType1'] = np.where(hp['BsmtFinType1'].isnull(),"NoBasement",hp['BsmtFinType1'])

hp['BsmtFinType2'] = np.where(hp['BsmtFinType2'].isnull(),"NoBasement",hp['BsmtFinType2'])

hp['Electrical'] = np.where(hp['Electrical'].isnull(),"SBrkr",hp['Electrical'])

hp['KitchenQual'] = np.where(hp['KitchenQual'].isnull(),"TA",hp['KitchenQual'])

hp['Functional'] = np.where(hp['Functional'].isnull(),"Typ",hp['Functional'])

hp['GarageType'] = np.where(hp['GarageType'].isnull(),"NoGarage",hp['GarageType'])

hp['GarageFinish'] = np.where(hp['GarageFinish'].isnull(),"NoGarage",hp['GarageFinish'])

hp['GarageQual'] = np.where(hp['GarageQual'].isnull(),"NoGarage",hp['GarageQual'])

hp['GarageCond'] = np.where(hp['GarageCond'].isnull(),"NoGarage",hp['GarageCond'])

hp['SaleType'] = np.where(hp['SaleType'].isnull(),"WD",hp['SaleType'])

hp['Utilities'] = np.where(hp['Utilities'].isnull(),"AllPub",hp['Utilities'])
# Lets check if all missing values are imputed for categorical column
cat_cols = hp.select_dtypes(['object']).columns
(hp[cat_cols].isnull().sum())*100/hp.shape[0]
hp.head()
# Some of the non-numeric predictors are stored as numbers; convert them into strings 
hp['MSSubClass'] = hp['MSSubClass'].apply(str)
hp['YrSold'] = hp['YrSold'].astype(str)
hp['MoSold'] = hp['MoSold'].astype(str)
def handle_missing(features):
    # the data description states that NA refers to typical ('Typ') values
    features['Functional'] = features['Functional'].fillna('Typ')
    # Replace the missing values in each of the columns below with their mode
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    
    # the data description stats that NA refers to "No Pool"
    features["PoolQC"] = features["PoolQC"].fillna("None")
    # Replacing the missing values with 0, since no garage = no cars in garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
        
    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # We have no particular intuition around how to fill in the rest of the categorical features
    # So we replace their missing values with None
    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))
        
    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int64', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))    
    return features

all_features = handle_missing(hp)
# Let's make sure we handled all the missing values
missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]
# Fetch all numeric features
numeric_dtypes = ['int64', 'float64']
numeric = []
for i in hp.columns:
    if hp[i].dtype in numeric_dtypes:
        numeric.append(i)
# Create bar charts for all numeric features
sns.set_style("white")
f, ax = plt.subplots(figsize=(20, 10))
ax.set_xscale("log")
ax = sns.barplot(data=hp[numeric],orient='h', palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
plt.xticks(rotation=90);
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
# Find skewed numerical features
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)
# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000
# Normalize skewed features
for i in skew_index:
    hp[i] = boxcox1p(hp[i], boxcox_normmax(hp[i] + 1))
# Let's make sure we handled all the skewed values
sns.set_style("white")
f, ax = plt.subplots(figsize=(20, 10))
ax.set_xscale("log")
ax = sns.barplot(data=hp[skew_index] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
hp['BsmtFinType1_Unf'] = 1*(hp['BsmtFinType1'] == 'Unf')
hp['HasWoodDeck'] = (hp['WoodDeckSF'] == 0) * 1
hp['HasOpenPorch'] = (hp['OpenPorchSF'] == 0) * 1
hp['HasEnclosedPorch'] = (hp['EnclosedPorch'] == 0) * 1
hp['Has3SsnPorch'] = (hp['3SsnPorch'] == 0) * 1
hp['HasScreenPorch'] = (hp['ScreenPorch'] == 0) * 1
hp['YearsSinceRemodel'] = hp['YrSold'].astype(int) - hp['YearRemodAdd'].astype(int)
hp['Total_Home_Quality'] = hp['OverallQual'] + all_features['OverallCond']
hp = hp.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
hp['TotalSF'] = hp['TotalBsmtSF'] + hp['1stFlrSF'] + hp['2ndFlrSF']
hp['YrBltAndRemod'] = hp['YearBuilt'] + hp['YearRemodAdd']

hp['Total_sqr_footage'] = (hp['BsmtFinSF1'] + hp['BsmtFinSF2'] +
                                hp['1stFlrSF'] + hp['2ndFlrSF'])
hp['Total_Bathrooms'] = (hp['FullBath'] + (0.5 * hp['HalfBath']) +
                               hp['BsmtFullBath'] + (0.5 * hp['BsmtHalfBath']))
hp['Total_porch_sf'] = (hp['OpenPorchSF'] + hp['3SsnPorch'] +
                              hp['EnclosedPorch'] + hp['ScreenPorch'] +
                              hp['WoodDeckSF'])
hp['TotalBsmtSF'] = hp['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
hp['2ndFlrSF'] = hp['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
hp['GarageArea'] = hp['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
hp['GarageCars'] = hp['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
hp['LotFrontage'] = hp['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
hp['MasVnrArea'] = hp['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
hp['BsmtFinSF1'] = hp['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

hp['haspool'] = hp['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
hp['has2ndfloor'] = hp['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
hp['hasgarage'] = hp['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
hp['hasbsmt'] = hp['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
hp['hasfireplace'] = hp['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res

log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']

hp = logs(hp, log_features)
def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res 

squared_features = ['YearRemodAdd', 'LotFrontage_log', 
              'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
              'GarageCars_log', 'GarageArea_log']
hp = squares(hp, squared_features)
hp = pd.get_dummies(hp).reset_index(drop=True)
hp.shape
hp.head()
# Remove any duplicated column names
hp = hp.loc[:,~hp.columns.duplicated()]
# Split features and labels
train_labels = hp_train['SalePrice'].reset_index(drop=True)
X = hp.iloc[:len(train_labels), :]
X_test = hp.iloc[len(train_labels):, :]
X.shape, train_labels.shape, X_test.shape
# Finding numeric features
numeric_dtypes = ['int64','float64']
numeric = []
for i in X.columns:
    if X[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms','Total_porch_sf','haspool','hasgarage','hasbsmt','hasfireplace']:
            pass
        else:
            numeric.append(i)     
# visualising some more outliers in the data values
fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(16, 180))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(X[numeric]), 1):
    if(feature=='MiscVal'):
        break
    plt.subplot(len(list(numeric)), 3, i)
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=hp_train)
        
    plt.xlabel('{}'.format(feature), size=15,labelpad=15)
    plt.ylabel('SalePrice', size=15, labelpad=15)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=15)
        plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='best', prop={'size': 10})
        
plt.show()
# Setup cross validation folds
kf = KFold(n_splits=10, random_state=50, shuffle=True)
# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)
# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=50)

# XGBoost Regressor
xgboost = XGBRegressor(objective='reg:squarederror',
                       learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=50)

# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=50)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=50)

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
scores = {}

score = cv_rmse(lightgbm)
print("lightgbm score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())
score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())
score = cv_rmse(ridge)
print("Ridge Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())
score = cv_rmse(svr)
print("SVR Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())
score = cv_rmse(rf)
print("Random Forest Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())
score = cv_rmse(gbr)
print("Gradient Boost Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())
print('stack_gen model is fitted')
stack_gen_model = stack_gen.fit(np.array(X), np.array(train_labels))
print('lightgbm model is fitted')
lgb_model_full_data = lightgbm.fit(X, train_labels)
print('Xgboost model is fitted')
xgb_model_full_data = xgboost.fit(X, train_labels)
print('Ridgde model is fitted')
ridge_model_full_data = ridge.fit(X, train_labels)
print('SVR model is fitted')
svr_model_full_data = svr.fit(X, train_labels)
print('Random Forest model is fitted')
rf_model_full_data = rf.fit(X, train_labels)
print('Gradient Boosting model is fitted')
gbr_model_full_data = gbr.fit(X, train_labels)
# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.1 * ridge_model_full_data.predict(X)) + \
            (0.2 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.1 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.05 * rf_model_full_data.predict(X)) + \
            (0.35 * stack_gen_model.predict(np.array(X))))
# Get final precitions from the blended model
blended_score = rmsle(train_labels, blended_predictions(X))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)
# Plot the predictions for each model
sns.set_style("white")
fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', size=20, labelpad=15)
plt.xlabel('Model', size=20, labelpad=15)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=15)

plt.title('Scores of Models', size=20)

plt.show()
