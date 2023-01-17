import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy.stats import norm, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Load the data 
df_train=pd.read_csv('/kaggle/input/house-prices-dataset/train.csv')
y_train = df_train.SalePrice.values
df_train.head()
df_test = pd.read_csv('/kaggle/input/house-prices-dataset/test.csv')
df_test.head()
df_train.info()
df_test.info()
#Size of the data
df_train.shape, df_test.shape
df_train.columns.shape
df_test.columns.shape
#since we are determining houseprices, lets explore the sale price
df_train['SalePrice'].describe()
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import cufflinks as cf
import plotly.offline as pyo
cf.go_offline()
pyo.init_notebook_mode()
print(__version__)
df_train['SalePrice'].iplot(kind='histogram', xTitle='price',yTitle='frequency',colors='darkred')
# Heatmap to show relationship between different variables
corr = df_train.corr()
plt.style.use('classic')
plt.subplots(figsize=(12,9))
sns.heatmap(corr, annot=False, vmax=0.9, square=True)
ad=12
cols=corr.nlargest(ad,'SalePrice')['SalePrice'].index  
cm=np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.30)
fig, ax = plt.subplots(figsize=(12, 9))
final_plot = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# Getting the skewness and kurtosis of the target variable
skew(df_train['SalePrice']), kurtosis(df_train['SalePrice'])
plt.figure(figsize=(6,3))
sns.distplot(df_train['SalePrice'])

fig = plt.figure(figsize=(6,3))
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()
print("Skewness: %f" %df_train['SalePrice'].skew())
print("Kurtosis: %f" %df_train['SalePrice'].kurt())
#Using the numpy fuction log1p which  applies log(1+x) to all to normalize the saleprice
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

#new distribution
plt.figure(figsize=(6,4))
sns.distplot(df_train['SalePrice'] , fit=norm);

#fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#QQ-plot
fig = plt.figure(figsize=(4,3))
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()
#Scatter plot for GrLivArea/saleprice
variable ='GrLivArea'
df_train.iplot(kind='scatter', x='GrLivArea', y='SalePrice',mode='markers',size=6, color='blue')
#scatter plot for TotalBsmtSF/saleprice
variable ='TotalBsmtSF'
df_train.iplot(kind='scatter', x='TotalBsmtSF', y='SalePrice',mode='markers',size=6, color='cyan')
#scatter plot for LotFrontage/saleprice
variable ='LotFrontage'
df_train.iplot(kind='scatter', x='LotFrontage', y='SalePrice',mode='markers',size=6, color='brown')
#Scatter plot for 1stFlrSF/saleprice
variable ='1stFlrSF'
df_train.iplot(kind='scatter', x='1stFlrSF', y='SalePrice',mode='markers',size=6, color='green')
variable ='SaleCondition'
fig, ax = plt.subplots(figsize=(14, 8))
fig = sns.boxplot(x=df_train.SaleCondition, y="SalePrice", data=df_train)
plt.xticks(rotation=60)
variable ='OverallQual'
fig, ax = plt.subplots(figsize=(14, 8))
fig = sns.boxplot(x=df_train.OverallQual, y="SalePrice", data=df_train)
plt.xticks(rotation=80)
#fig.axis(ymin=0, ymax=8000)
#scatter plots between SalePrice and correlated variables
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath',
        'YearBuilt','Fireplaces','YearRemodAdd']
sns.pairplot(df_train[cols], size = 3.5)
plt.show()
df_train.isna().sum()
df_train.isna().sum().value_counts()
df_train["PoolQC"] = df_train["PoolQC"].fillna("None")

df_train["MiscFeature"] = df_train["MiscFeature"].fillna("None")

df_train["Alley"] = df_train["Alley"].fillna("None")

df_train["Fence"] = df_train["Fence"].fillna("None")

df_train["FireplaceQu"] = df_train["FireplaceQu"].fillna("None")

df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df_train[col] = df_train[col].fillna('None')


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df_train[col] = df_train[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df_train[col] = df_train[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_train[col] = df_train[col].fillna('None')

df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")
df_train["MasVnrArea"] = df_train["MasVnrArea"].fillna(0)

df_train['MSZoning'] = df_train['MSZoning'].fillna(df_train['MSZoning'].mode()[0])

df_train = df_train.drop(['Utilities'], axis=1)

df_train["Functional"] = df_train["Functional"].fillna("Typ")

df_train['Electrical'] = df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])

df_train['KitchenQual'] = df_train['KitchenQual'].fillna(df_train['KitchenQual'].mode()[0])

df_train['Exterior1st'] = df_train['Exterior1st'].fillna(df_train['Exterior1st'].mode()[0])
df_train['Exterior2nd'] = df_train['Exterior2nd'].fillna(df_train['Exterior2nd'].mode()[0])

df_train['SaleType'] = df_train['SaleType'].fillna(df_train['SaleType'].mode()[0])

df_train['MSSubClass'] = df_train['MSSubClass'].fillna("None")

df_test["PoolQC"] = df_test["PoolQC"].fillna("None")

df_test["MiscFeature"] = df_test["MiscFeature"].fillna("None")

df_test["Alley"] = df_test["Alley"].fillna("None")

df_test["Fence"] = df_test["Fence"].fillna("None")

df_test["FireplaceQu"] = df_test["FireplaceQu"].fillna("None")

df_test["LotFrontage"] = df_test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df_test[col] = df_test[col].fillna('None')


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df_test[col] = df_test[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df_test[col] = df_test[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_test[col] = df_test[col].fillna('None')

df_test["MasVnrType"] = df_test["MasVnrType"].fillna("None")
df_test["MasVnrArea"] = df_test["MasVnrArea"].fillna(0)

df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])

df_test = df_test.drop(['Utilities'], axis=1)

df_test["Functional"] = df_test["Functional"].fillna("Typ")

df_test['Electrical'] = df_test['Electrical'].fillna(df_test['Electrical'].mode()[0])

df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])

df_test['Exterior1st'] = df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])
df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])

df_test['SaleType'] = df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])

df_test['MSSubClass'] = df_test['MSSubClass'].fillna("None")
#More feature engineering to transform some numerical variables that are really categorical
adrian = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']
for i in adrian:
    df_train[i] = df_train[i].apply(str)
    
for j in adrian:
    df_test[j] = df_test[j].apply(str)
#Label Encoding
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for j in cols:
    encoder = LabelEncoder() 
    encoder.fit(list(df_train[j].values)) 
    df_train[j] = encoder.transform(list(df_train[j].values))
    
for k in cols: 
    encoder = LabelEncoder()
    encoder.fit(list(df_test[k].values)) 
    df_test[k] = encoder.transform(list(df_test[k].values))
#Shape of the data
df_train.shape, df_test.shape
numerical_features = df_train.dtypes[df_train.dtypes != "object"].index

# Checking the skewness of all numerical features
skewed_features = df_train[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_features})
print(skewness.head())

numerical_featurez = df_test.dtypes[df_test.dtypes != "object"].index

# Checking the skewness of all numerical features
skewed_featurez = df_test[numerical_featurez].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical featurez: \n")
skewnes = pd.DataFrame({'Skew' :skewed_featurez})
print(skewnes.head())
skewness = skewness[abs(skewness) > 1.0]
print("There are {} skewed numerical features to boxcox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lmbda = 0.15
for i in skewed_features:
    df_train[i] = boxcox1p(df_train[i], lmbda)
    
    
skewnes = skewnes[abs(skewnes) > 1.0]
print("There are {} skewed numerical features to boxcox transform".format(skewnes.shape[0]))

skewed_featurez = skewnes.index
lmbda = 0.15
for i in skewed_featurez:
    df_test[i] = boxcox1p(df_test[i], lmbda)
train_set = df_train.shape[0]
test_set = df_test.shape[0]
data = pd.concat((df_train, df_test)).reset_index(drop=True)
data.drop(['SalePrice', 'Id'], axis=1, inplace=True)
print("data size is : {}".format(data.shape))
data = pd.get_dummies(data)
print(data.shape)
#data = data.sample(frac=1.0)
#data
train = data[:train_set]
test = data[train_set:]
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
Enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)).fit(train, y_train)
Enet
Ridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5).fit(train,y_train)
Ridge
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)).fit(train, y_train)
lasso
Xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                            learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1).fit(train, y_train)
Xgb
Lgbm = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=700,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11).fit(train, y_train)
Lgbm
Gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5).fit(train,y_train)
Gboost
#Scores for different models
print('LGBM score : {:.4f}'.format(Lgbm.score(train, y_train)))
print('Xgb score : {:.4f}'.format(Xgb.score(train, y_train)))
print('Enet score : {:.4f}'.format(Enet.score(train, y_train)))
print('Lasso score : {:.4f}'.format(lasso.score(train, y_train)))
print('Ridge score : {:.4f}'.format(Ridge.score(train, y_train)))
print('Gboost score : {:.4f}'.format(Gboost.score(train, y_train)))
lasso_pred = (lasso.predict(test))
Ridge_pred = (Ridge.predict(test))
Xgb_pred = (Xgb.predict(test))
Lgbm_pred = (Lgbm.predict(test))
Enet_pred = (Enet.predict(test))
Gboost_pred = (Gboost.predict(test))
# Getting final predictions 
final_prediction = Xgb_pred*0.35 + Lgbm_pred*0.20 + Gboost_pred*0.20 + lasso_pred*0.10 + Ridge_pred*0.10 + Enet_pred*0.05
final_prediction 
test_data = pd.read_csv('/kaggle/input/house-prices-dataset/test.csv')
test_data
#Final submission
submission = pd.DataFrame()
submission['Id'] = test_data['Id']
submission['SalePrice'] = final_prediction
submission
submission.to_csv('submission.csv', index=False)