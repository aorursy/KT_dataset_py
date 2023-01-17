# Importing packages for EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:20,.2f}'.format
# Import datasets
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col=0)
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
print(f'Size of training set : {train_data.shape[0]}')
print(f'Size of evaluation set : {test_data.shape[0]}')
print(f'No. of columns including target variable : {train_data.shape[1]}')
train_data.head()
test_data.head()
X1 = train_data.drop('SalePrice',axis=1)
X2 = test_data
X = X1.append(X2)
y = train_data['SalePrice']
y.describe()
# Visualizing distribution
fig = plt.figure(figsize=(8,6));
plt.ticklabel_format(style='plain');
plt.title('Distribution of SalePrice');
sns.distplot(y);
print(f'Skewness of SalePrice : {y.skew():.4f}')
print(f'Kurtosis of SalePrice : {y.kurt():.4f}')
# Using log(1+y) transformation on SalePrice
logy = np.log1p(y)    # New target variable
train_data['LogSalePrice'] = logy
fig = plt.figure(figsize=(8,6));
plt.title('Distribution of SalePrice after log transformation');
sns.distplot(logy); 
print(f'No. of numerical variables : {train_data.select_dtypes(include= [np.number]).shape[1]}')
print(f'No. of categorical variables : {train_data.select_dtypes(include= [np.object]).shape[1]}')
# Summary of numerical variables
X.describe(include=[np.number]).T
# Heatmap of top 10 features correlated to SalePrice
fig = plt.figure(figsize=(12,10))
top_cols = X1.corrwith(train_data['SalePrice']).nlargest(n=10).index.to_list()
top_cols.append('SalePrice')
sns.heatmap(train_data[top_cols].corr(), linewidths=0.2, annot=True, cmap="Blues");
# Pairplot to explore relationship between the areas.
sns.pairplot(X1[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']]);
# Heatmap to show correlations
sns.heatmap(train_data[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'SalePrice']].corr(), linewidths=0.2, annot=True, cmap="OrRd");
# check for missing values
X[['TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea']].isnull().sum()
X['TotalBsmtSF'].fillna(0, inplace=True) #indicates no basement
X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF'] + X['GrLivArea']
X1['TotalSF'] = train_data['TotalSF'] = X['TotalSF'].iloc[:X1.shape[0]]
fig = plt.figure(figsize=(16,12));
fig.subplots_adjust(hspace=0.3, wspace=0.2);
ax1 = fig.add_subplot(221);
cor = X1['TotalBsmtSF'].corr(y);
ax1.title.set_text(f'Correlation : {cor}');
sns.regplot(x='TotalBsmtSF', y='SalePrice', data=train_data);
ax2 = fig.add_subplot(222);
cor = X1['GrLivArea'].corr(y);
ax2.title.set_text(f'Correlation : {cor}');
sns.regplot(x='GrLivArea', y='SalePrice', data=train_data);
ax3 = fig.add_subplot(223);
cor = X1['1stFlrSF'].corr(y);
ax3.title.set_text(f'Correlation : {cor}');
sns.regplot(x='1stFlrSF', y='SalePrice', data=train_data);
ax4 = fig.add_subplot(224);
cor = X1['2ndFlrSF'].corr(y);
ax4.title.set_text(f'Correlation : {cor}');
sns.regplot(x='2ndFlrSF', y='SalePrice', data=train_data);
fig = plt.figure(figsize=(8,6));
cor = X1['TotalSF'].corr(y)
plt.title(f'Correlation : {cor}');
sns.regplot(x='TotalSF', y='SalePrice', data=train_data);
fig = plt.figure(figsize=(16,12));
fig.subplots_adjust(hspace=0.3, wspace=0.2);
ax1 = fig.add_subplot(221);
cor = X1['OpenPorchSF'].corr(y);
ax1.title.set_text(f'Correlation : {cor}');
sns.regplot(x='OpenPorchSF', y='SalePrice', data=train_data);
ax2 = fig.add_subplot(222);
cor = X1['EnclosedPorch'].corr(y);
ax2.title.set_text(f'Correlation : {cor}');
sns.regplot(x='EnclosedPorch', y='SalePrice', data=train_data);
ax3 = fig.add_subplot(223);
cor = X1['3SsnPorch'].corr(y);
ax3.title.set_text(f'Correlation : {cor}');
sns.regplot(x='3SsnPorch', y='SalePrice', data=train_data);
ax4 = fig.add_subplot(224);
cor = X1['ScreenPorch'].corr(y);
ax4.title.set_text(f'Correlation : {cor}');
sns.regplot(x='ScreenPorch', y='SalePrice', data=train_data);
X['TotalPorchSF'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']
X1['TotalPorchSF'] = train_data['TotalPorchSF'] = X['TotalPorchSF'].iloc[:X1.shape[0]]
fig = plt.figure(figsize=(8,6));
cor = X1['TotalPorchSF'].corr(y)
plt.title(f'Correlation : {cor}');
sns.regplot(x='TotalPorchSF', y='SalePrice', data=train_data);
X.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea'], axis=1, inplace=True)
X.drop(['TotalPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], axis=1, inplace=True)
# New feature: Age of property at the time it was bought
X['HouseAge'] = X['YrSold'] - X['YearBuilt']
X.HouseAge[X.HouseAge < 0] = 0
X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
X.RemodAge[X.RemodAge < 0] = 0
# Dropping the year and month columns
X.drop(['GarageYrBlt', 'YrSold', 'YearBuilt', 'MoSold', 'YearRemodAdd'], axis=1, inplace=True)
fig = plt.figure(figsize=(8,6));
sns.scatterplot(x='HouseAge', y='SalePrice', data=X.join(y));
fig = plt.figure(figsize=(8,6));
sns.scatterplot(x='RemodAge', y='SalePrice', data=X.join(y));
# summary of categorical variables
X.describe(include=[np.object]).T

fig = plt.figure(figsize=(16,12));
fig.subplots_adjust(hspace=0.3, wspace=0.2);
ax1 = fig.add_subplot(221);
sns.boxplot(x='OverallQual', y='SalePrice', data=train_data);
ax2 = fig.add_subplot(222);
sns.boxplot(x='OverallCond', y='SalePrice', data=train_data);
ax3 = fig.add_subplot(223);
sns.boxplot(x='GarageQual', y='SalePrice', data=train_data);
ax4 = fig.add_subplot(224);
sns.boxplot(x='GarageCond', y='SalePrice', data=train_data);
fig = plt.figure(figsize=(16,12));
fig.subplots_adjust(hspace=0.3, wspace=0.2);
ax1 = fig.add_subplot(221);
sns.boxplot(x='MasVnrType', y='SalePrice', data=train_data);
ax2 = fig.add_subplot(222);
sns.boxplot(x='BsmtFinType1', y='SalePrice', data=train_data);
ax3 = fig.add_subplot(223);
sns.boxplot(x='BsmtQual', y='SalePrice', data=train_data);
ax4 = fig.add_subplot(224);
sns.boxplot(x='SaleCondition', y='SalePrice', data=train_data);
fig = plt.figure(figsize=(20,6));
plt.xticks(rotation=45);
sns.boxplot(x='Neighborhood', y='SalePrice', data=train_data);
# Missing values
missing_values = X.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(inplace=True)
missing_values = missing_values.to_frame()
missing_values.columns = ['count']
missing_values['Name'] = missing_values.index
missing_values = missing_values.reset_index(drop=True)
missing_values
fig = plt.figure(figsize=(20,6));
missing_values = missing_values.sort_values('count', ascending=False)
sns.barplot(x = 'Name', y = 'count', data=missing_values);
plt.xticks(rotation = 90);
plt.show();
X['PoolQC'].value_counts()
# all other columns are mapped into 1 to reduce disparity
# 1 indicates presence of pool and 0 indicates no pool
pool_map = {'Ex':1, 'Gd':1, 'Fa':1}
X['PoolQC'] = X['PoolQC'].fillna(0).replace(pool_map)
X['PoolQC'].value_counts()
X['MiscFeature'].fillna('NA', inplace=True)     # No misc feature
X['Alley'].fillna('NA', inplace=True)          # No alley access
X['Fence'].fillna('NA', inplace=True)           # No fence
X['FireplaceQu'].fillna('NA', inplace=True)     # No fireplace
X['MasVnrType'].fillna('NA', inplace=True)    # No Masonry Veneer
X['MasVnrArea'].fillna(0, inplace=True)         # No Masonry Veneer
for col in ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']:
    X[col].fillna('NA', inplace=True)           # No Basement
X['Functional'].fillna('Typ', inplace=True)     # Typical
X['BsmtHalfBath'].fillna(0, inplace=True)       # No Basement half bathrooms
X['BsmtFullBath'].fillna(0, inplace=True)       # No Basement full bathrooms
for col in ['GarageType','GarageQual','GarageCond','GarageFinish']:
    X[col].fillna('NA', inplace=True)         # No Garage
X['GarageArea'].fillna(0, inplace=True)         # No Garage
X['GarageCars'].fillna(0, inplace=True)         # No Garage
for col in ['BsmtUnfSF','BsmtFinSF1','BsmtFinSF2']:
    X[col].fillna(0, inplace=True)  # Finished or no Basements
X['SaleType'].fillna('Oth', inplace=True)       # Other
X['Exterior1st'].fillna('Other', inplace=True)  # Other
X['Exterior2nd'].fillna(X['Exterior1st'], inplace=True) # Assume same exterior material
X['Utilities'].fillna(X['Utilities'].mode()[0], inplace=True)
X['KitchenQual'].fillna(X['KitchenQual'].mode()[0], inplace=True)
X['Electrical'].fillna(X['Electrical'].mode()[0], inplace=True)
X['MSZoning'].fillna(X['MSZoning'].mode()[0], inplace=True)
X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median())) # Filling with median based on neighborhood

X.isnull().sum().sum()
# Based on description, MSSubClass is a categorical variable 
X['MSSubClass'] = X['MSSubClass'].astype(str)
X['MSZoning'].value_counts()
X['MSZoning'] = X['MSZoning'].replace('C (all)', 'C')
# Ordinal encoding
ordinal_map = {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
for col in ['ExterQual','ExterCond','HeatingQC','KitchenQual','BsmtQual','BsmtCond','FireplaceQu','GarageQual','GarageCond']:
    X[col] = X[col].map(ordinal_map)
# Binary encoding
X['CentralAir'] = X['CentralAir'].map({'N':0, 'Y':1})
X['Street'] = X['Street'].map({'Grvl':0, 'Pave':1})

# select nominal variables
X_cat_cols = list(X.select_dtypes(include=[np.object]).columns)
X_num = X.drop(X_cat_cols, axis=1)
X1_num = X_num.head(train_data.shape[0])
y = logy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
model = LinearRegression()
# Function to scale features
def scale_data(x):
    scaler = MinMaxScaler()
    temp = scaler.fit_transform(x)
    x.loc[:,:] = temp
    return x
print(f'No. of numeric features : {X_num.shape[1]}')
from sklearn.feature_selection import RFE
# Select optimal no. of features
l = np.arange(1,42)
max_score = 0
score_list = []
n = 0
for i in range(len(l)):
    X_train, X_val, y_train, y_val = train_test_split(scale_data(X1_num), y, test_size=0.2, random_state=0)
    rfe = RFE(model, l[i])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_val_rfe = rfe.transform(X_val)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_val_rfe,y_val)
    score_list.append(score)
    if score > max_score:
        max_score = score
        n = l[i]

print("Score with %d features: %f" % (n, max_score))

model = LinearRegression()
rfe = RFE(model, n)
X1_rfe = rfe.fit_transform(X1_num,y)
temp = pd.Series(rfe.support_,index = X1_num.columns)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
X_num = X[selected_features_rfe]
X1_cat = X1[X_cat_cols]
X1_cat.head()
# Total no. unique values in each variable
X1_cat.nunique().sort_values(ascending=False)
# Dropping values with too many unique features (>10)
X1_cat.drop(['Neighborhood','Exterior1st','Exterior2nd','MSSubClass'], axis=1, inplace=True)
X_cat = X[X1_cat.columns]
X_cat = pd.get_dummies(X_cat, drop_first=True)
# Check if feature matrix is sparse
from scipy.sparse import issparse
issparse(X)
from sklearn.model_selection import train_test_split
X = X_num.join(X_cat)
X_tv = X.head(train_data.shape[0])
X_test = X.tail(test_data.shape[0])
X_train, X_val, y_train, y_val = train_test_split(X_tv, y, test_size=0.2)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
# Regresssion models
lasso_reg = make_pipeline(MinMaxScaler(),Lasso(alpha=0.0005, random_state=1))
ridge_reg = make_pipeline(MinMaxScaler(),Ridge(alpha=0.0005, random_state=1))
elasticnet_reg = make_pipeline(MinMaxScaler(),ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
rf_reg = RandomForestRegressor()
xgb_reg = XGBRegressor()
lgbm_reg = LGBMRegressor()
# K-fold cross validation using root mean squared errors
from sklearn.model_selection import cross_val_score, KFold
def rmsle_cv(model):
    kf = KFold(10, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
score = rmsle_cv(lasso_reg)
print("Lasso regression rmse mean & std: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ridge_reg)
print("Ridge regression rmse mean & std: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(elasticnet_reg)
print("ElasticNet regression rmse mean & std: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(rf_reg)
print("Random forest regression rmse mean & std: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(xgb_reg)
print("XGBoost regression rmse mean & std: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(lgbm_reg)
print("LightGBM regression rmse mean & std: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# selected model
model = lgbm_reg
# LightGBM has the least rmse mean
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print(f'R squared value : {model.score(X_val, y_val)}')
fig = plt.figure(figsize=(8,6))
ax = sns.scatterplot(y_pred, y_val);
ax.set(xlabel = 'Predicted values', ylabel = 'Actual values');

from sklearn.metrics import mean_absolute_error, mean_squared_error
print(f'Mean squared error on validation set : {mean_squared_error(y_pred, y_val):.3f}')
print(f'Mean absolute error on validation set : {mean_absolute_error(y_pred, y_val):.3f}')
# Fitting entire data set to the model
model.fit(X_tv,y)
# submission
sub = pd.DataFrame()
sub['Id'] = X_test.index
sub['SalePrice'] = np.expm1(model.predict(X_test)) # To invert the log(1+x) function
sub.to_csv('submission.csv',index=False)
sub.head()
