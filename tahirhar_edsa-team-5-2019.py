import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import Lasso, Ridge, LinearRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error



import xgboost as xgb



from scipy import stats

from scipy.stats import norm, skew #for some statistics

from scipy.special import boxcox1p

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

test2 = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

len_train = train.shape[0]

## concatenate the two datasets

combined = pd.concat([train,test], sort=False)
print(train.shape)

print(test.shape)
#Checking data types

combined.select_dtypes(include='object').head()
# Numerical feautures graphs



num_graphs = list(train[list(train.describe().columns)[1:]].corr()['SalePrice'].sort_values(ascending=False).reset_index()['index'].values)[1:]



plt.subplots(figsize=(15, 30))

for i, col in enumerate(num_graphs):

    ax = plt.subplot(9, 4, i+1)    

    sns.regplot(x = col,y = 'SalePrice', data = train[train.columns[1:]], ax=ax)

    plt.title(col+' vs '+'SalePrice'+' : '+str(round(train.corr()['SalePrice'][col]*100,2))+'%');

    

plt.tight_layout()
#Setting categorical column with numerical values to string

combined['MSSubClass'] = combined['MSSubClass'].astype(str)
#Check numerical data types

combined.dtypes[combined.dtypes != "object"].head()
combined.select_dtypes(include='object').isnull().sum()[combined.select_dtypes(include='object').isnull().sum()>0]
# View where training data is missing 

missing_data = combined.isnull().sum()

missing_data = missing_data[missing_data > 0]

missing_data.sort_values(inplace=True)

missing_data.plot.bar(title='Missing Combined Data');
#Based on data descriptions the following columns NA's will be filled with 'None'

none_columns = ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

           'PoolQC','Fence','MiscFeature')

#Based on data descriptions the following columns NA's will be filled with mode

mode_columns = ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional')
#Fill categorical NA's

for col in none_columns:

    train[col]=train[col].fillna('None')

    test[col]=test[col].fillna('None')

    

for col in mode_columns:

    train[col]=train[col].fillna(train[col].mode()[0])

    test[col]=test[col].fillna(train[col].mode()[0])
#Based on data descriptions the following numerical columns will be filled with 0

zeros = ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea')



for col in zeros:

    train[col] = train[col].fillna(0)

    test[col] = test[col].fillna(0)
#fill Lotfrontage with average of the neighbourhood

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
train = train.drop('Utilities', 1)

test = test.drop('Utilities', 1)
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.grid(color='grey', linestyle=':', linewidth=1)

plt.show()
#removing outliers recomended by author, but less than $300 000

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'], c='grey')

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.grid(color='grey', linestyle=':', linewidth=1)

plt.show()
(mu, sigma) = norm.fit(train['SalePrice'])

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train['SalePrice'],fit=norm)

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1, 2, 2)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')

quantile_plot=stats.probplot(train['SalePrice'], plot=plt)
# Fix the Skewness using 'nplog'

train["SalePrice"] = np.log1p(train["SalePrice"])
(mu, sigma) = norm.fit(train['SalePrice'])

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train['SalePrice'],fit=norm)

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1, 2, 2)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')

quantile_plot=stats.probplot(train['SalePrice'], plot=plt)
#Set the length of the train and concat once again to create the combined DF

len_train=train.shape[0]

print(train.shape)

combined = pd.concat([train,test], sort=False)
# Adding Features

combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']
# check for NAs once more

combined.select_dtypes(include=['int','float','int64']).isnull().sum()[combined.select_dtypes(include=['int','float','int64']).isnull().sum()>0]
#Label Encoding



from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(combined[c].values)) 

    combined[c] = lbl.transform(list(combined[c].values))



# shape        

print('Shape combined: {}'.format(combined.shape))
combined.head()
combined.select_dtypes(include=['int','int64','float']).columns
num_feats = combined.dtypes[combined.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = combined[num_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness.drop('SalePrice', 0)

skewness.head(15)
#Correct for skewness by using boxcox1p

skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    combined[feat] = boxcox1p(combined[feat], lam)
#Get Dummies

combined = pd.get_dummies(combined)

len_train = train.shape[0]
train.shape
train = combined[:len_train]

y_train = train.SalePrice.values

train = train.drop(['Id','SalePrice'], 1)

test = combined[len_train:]

test = test.drop(['Id','SalePrice'], 1)

train.shape
#Validation function - Courtesy of ....

n_folds = 5

def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)

LinReg = LinearRegression()

score = rmsle_cv(LinReg)

print("Random Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
RG = Ridge(alpha=22)

score = rmsle_cv(RG)

print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005))

score = rmsle_cv(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
RandomForest = RandomForestRegressor(n_estimators=150, random_state = 1)

score = rmsle_cv(RandomForest)

print("Random Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
data_dmatrix = xgb.DMatrix(data=train.values,label=y_train)

XGBoost = xgb.XGBRegressor(random_state = 5, max_depth = 2, alpha = 10, n_estimators = 1000,

                           learning_rate = 0.05, objective = 'reg:squarederror', colsample_bytree = 0.1,

                          subsample = 0.75)

score = rmsle_cv(XGBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#Fit the training dataset on every model

#lr = LinReg.fit(train, y_train)

ls = lasso.fit(train, y_train)

rg = RG.fit(train, y_train)

#rf = RandomForest.fit(train, y_train)

gb = GBoost.fit(train, y_train)

xg = XGBoost.fit(train,y_train)
pred_ls = np.expm1(ls.predict(test))

pred_rg = np.expm1(rg.predict(test))

pred_gb = np.expm1(gb.predict(test))

pred_xg = np.expm1(xg.predict(test))
final_predictions = (pred_ls + pred_rg + pred_gb + pred_xg) / 4

# Tried weighted average, scored less

#final_weighted = (0.25 * pred_ls) + (0.30 * pred_rg) + (0.20 * pred_gb) + (0.25 * pred_xg)
final_predictions
#Output to CSV

output_avg = pd.DataFrame({'Id':test2.Id, 'SalePrice': final_predictions})

output_avg.to_csv('submission.csv', index=False)