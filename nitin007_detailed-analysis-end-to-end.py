# Imports

import numpy as np

import pandas as pd

from pandas import DataFrame

from matplotlib import pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
house_data = pd.read_csv("../input/train.csv")

house_data.head(10)
house_data.shape
house_data.describe()
new_df = house_data._get_numeric_data()

new_df.shape
pd.set_option('display.max_columns',None) #to show all the columns in a dataframe

new_df.describe()
size = 38

cols = new_df.columns

data_corr = new_df.corr()

threshold = 0.4

corr_list = []
for i in range(0, size):

    for j in range(i+1, size):

        if (data_corr.iloc[i,j]>=threshold and data_corr.iloc[i,j]<1 or\

           data_corr.iloc[i,j]<=0 and data_corr.iloc[i,j]<=-threshold):

            corr_list.append([data_corr.iloc[i,j],i,j])

            

# sorted list to show the highest values first

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))
# print the sorted list

for i,j,k in s_corr_list:

    if cols[k]!= 'SalePrice':

        print ("%s and %s = %0.2f"%(cols[j], cols[k],i))
# print the sorted list (Correlation with Sale Price)

for i,j,k in s_corr_list:

    if cols[k]=='SalePrice':

        print ("%s and %s = %0.2f"%(cols[j], cols[k],i))

    
f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(data_corr,vmax=0.8,square=True);
sns.distplot(house_data['SalePrice'],kde=True,color='b', hist_kws={'alpha':0.5})
house_data['SalePrice'].describe()
sns.regplot(x='OverallQual',y='SalePrice',color='green',data=house_data)
cols = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF',\

        'FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt', 'MasVnrArea',\

        'Fireplaces']



for i in cols:

    fig = plt.figure(figsize=(8,6))

    plt.scatter(house_data[i],house_data['SalePrice'],color='b')

    plt.xlabel(i,fontsize=14)

    plt.ylabel('SalePrice', fontsize=14)

    plt.show()
c_data = house_data.select_dtypes(include=['object'])
cols = c_data.columns

house_data['YrSold'].head()
fig = plt.figure(figsize=(12,6))

sns.boxplot(x='Neighborhood',y=house_data['SalePrice'],data=c_data)

plt.xticks(rotation = 45);
fig = plt.figure(figsize=(12,6))

sns.countplot(x='Neighborhood', data=c_data)

plt.xticks(rotation=45);
fig,ax = plt.subplots(2,1,figsize=(10,6))

sns.boxplot(x='SaleType',y=house_data['SalePrice'],data=c_data,ax=ax[0])

sns.boxplot(x='SaleCondition',y=house_data['SalePrice'], data= c_data, ax=ax[1])

plt.tight_layout()
new_data = c_data.drop(['Street','Alley','Utilities','LandSlope','PavedDrive','SaleType',\

                        'SaleCondition'],axis=1)
g = sns.FacetGrid(house_data,col='YrSold', col_wrap=3)

g.map(sns.boxplot,'MoSold','SalePrice', palette ='Set2',order=range(1,13)).set(ylim=(0,500000))

plt.tight_layout()
fig,ax = plt.subplots(2,1,figsize=(10,8))

sns.boxplot(x='BldgType', y= house_data['SalePrice'], data=c_data,ax=ax[0])

sns.boxplot(x='HouseStyle', y= house_data['SalePrice'], data= c_data, ax=ax[1])

plt.xticks(rotation=45)

plt.show()
fig,ax = plt.subplots(2,1,figsize=(10,8))

sns.boxplot(x='Condition1', y= house_data['SalePrice'], data= c_data, ax = ax[0] )

sns.boxplot(x='Exterior1st', y= house_data['SalePrice'], data=c_data, ax=ax[1])

plt.xticks(rotation=45)

plt.show()
fig,ax = plt.subplots(2,2,figsize=(10,8))

sns.boxplot(x='BsmtQual', y=house_data['SalePrice'], data=c_data, ax=ax[0,0])

sns.boxplot(x='BsmtCond', y=house_data['SalePrice'], data=c_data, ax=ax[0,1])

sns.boxplot(x='BsmtExposure', y=house_data['SalePrice'], data=c_data, ax=ax[1,0])

sns.boxplot(x='BsmtFinType1', y=house_data['SalePrice'], data=c_data, ax=ax[1,1])

plt.tight_layout()
sns.factorplot('FireplaceQu', 'SalePrice', data= house_data, estimator=np.median,\

              order = ['Ex','Gd','TA','Fa','Po'], size=5, aspect=1.3)
pd.crosstab(house_data.FireplaceQu, house_data.Fireplaces)
g = sns.FacetGrid(house_data,col='FireplaceQu', col_wrap=3,palette='Set2',col_order= ['Ex','Gd','TA','Fa','Po'])

g.map(sns.boxplot,'Fireplaces','SalePrice', order=[1,2,3],palette='Set2')

plt.tight_layout()
pd.crosstab(house_data.HeatingQC, house_data.CentralAir)
sns.factorplot('HeatingQC','SalePrice',hue='CentralAir',estimator=np.mean,data=house_data,size= 5,aspect=1.3)
sns.factorplot('KitchenQual','SalePrice',data=house_data,estimator=np.median,size=5,aspect=1.3\

              ,order=['Ex','Gd','TA','Fa'])
sns.boxplot(x='MSZoning',y = 'SalePrice',data=house_data)
total = house_data.isnull().sum().sort_values(ascending=False)

percent =(total/house_data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent],axis=1,keys=['Total', 'Percent'])

missing_data.sort_values('Total', ascending=False).head(20)
fig, ax = plt.subplots(2,2,figsize=(10,6))

sns.boxplot(x='GarageType', y='SalePrice', data=house_data, ax = ax[0,0])

sns.boxplot(x='GarageCond', y='SalePrice', data=house_data, ax = ax[1,0])

sns.boxplot(x='GarageFinish', y='SalePrice', data=house_data, ax = ax[0,1])

sns.boxplot(x='GarageQual', y='SalePrice', data=house_data, ax = ax[1,1])

plt.tight_layout()
sns.regplot('GarageYrBlt', 'SalePrice', data=house_data)

plt.show()
sns.factorplot(x = 'MSSubClass', y = 'SalePrice', data=house_data, size=5, estimator=np.median)

plt.show()
# Lot shape

sns.boxplot(x = 'LotShape', y='SalePrice', data=house_data)

plt.show()
# LotFrontage

plt.scatter(house_data['LotFrontage'], house_data['SalePrice'])

plt.show()
# LotArea

plt.scatter(house_data['LotArea'], house_data['SalePrice'])

plt.show()
sns.boxplot(x = 'LotConfig', y='SalePrice', data=house_data)

plt.show()
house_data.LotArea.isnull().sum() # 0 missing

house_data.LotConfig.isnull().sum() # 0 missing

house_data.LotFrontage.isnull().sum() # 258 values missing (Approx 17%)
house_data.LotShape.isnull().sum() # 0 missing
# LandContour

sns.boxplot(x='LandContour', y='SalePrice', data=house_data)

plt.show()
# LandSlope

sns.boxplot(x='LandSlope', y='SalePrice', data=house_data)

plt.show()
# MasVnrArea

plt.scatter(house_data['MasVnrArea'], house_data['SalePrice'])

plt.show()
house_data.MasVnrArea.isnull().sum()
house_data.MasVnrArea.value_counts()
# MasVnrType

sns.boxplot(x='MasVnrType', y='SalePrice', data=house_data)

plt.show()
house_data.MasVnrType.isnull().sum()
house_data.MasVnrType.value_counts()
house_data['BsmtQual'].value_counts()
# variables to keep

numerical = ['MSSubClass','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF',\

             '1stFlrSF','FullBath','TotRmsAbvGrd','Fireplaces','Neighborhood',\

             'LotArea','LotFrontage']

categorical = ['Neighborhood', 'BsmtQual','HeatingQC', 'CentralAir','KitchenQual', 'Electrical']
df_train = house_data[['Id','MSSubClass','OverallQual','GrLivArea','GarageCars','GarageArea',\

                       'TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd', 'Fireplaces',\

                       'Neighborhood','BsmtQual','HeatingQC','CentralAir','KitchenQual', \

                       'Electrical','SalePrice','LotArea','LotFrontage']]
df_train.select_dtypes(include=[np.number]).skew()
df_test = pd.read_csv('../input/test.csv')

df_test = df_test[['Id','MSSubClass','OverallQual','GrLivArea','GarageCars','GarageArea',\

                   'TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd', 'Fireplaces',\

                   'Neighborhood','BsmtQual','HeatingQC','CentralAir','KitchenQual',\

                   'Electrical','LotArea','LotFrontage']]



df_test.describe()
def impute(column, value):

    df_train.loc[df_train[column].isnull(), column] = value

    

def impute_test(column, value):

    df_test.loc[df_test[column].isnull(), column] = value
# impute the most common value

impute('Electrical', 'SBrkr')

impute('BsmtQual', 'TA')

impute('LotFrontage', df_train['LotFrontage'].median())
# Imputing for Test data

cols_test = ['GarageCars', 'GarageArea', 'TotalBsmtSF']

for i in cols_test:

    impute_test(i, np.mean(df_test[i]))

df_test.describe()
from sklearn.preprocessing import StandardScaler

Saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = Saleprice_scaled[Saleprice_scaled[:,0].argsort()][:10]

high_range = Saleprice_scaled[Saleprice_scaled[:,0].argsort()][-10:]

print ('outer range(low) of the distribution:')

print (low_range)

print('\nouter range(high) of the distribution:')

print(high_range)
cols = df_train.select_dtypes(include=[np.number])

for i in cols:

    if i != 'SalePrice':

        plt.scatter(df_train[i], df_train['SalePrice'], data=df_train)

        plt.xlabel(i)

        plt.ylabel('SalePrice')

        plt.show()
df_train.sort_values(by = 'GrLivArea', ascending=False)[:2] # last two values seems to be outliers

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
df_train.sort_values(by = 'TotalBsmtSF', ascending= False)[:1] # last value as clear from the scatterplot above

df_train = df_train.drop(df_train[df_train['Id'] == 333].index)
df_train.sort_values(by = '1stFlrSF', ascending= False)[:1] # last value as clear from the scatterplot above

df_train = df_train.drop(df_train[df_train['Id'] == 497].index)
df_train.sort_values(by = 'TotRmsAbvGrd', ascending= False)[:1] # last value as clear from the scatterplot above

df_train = df_train.drop(df_train[df_train['Id'] == 636].index)
df_train.sort_values(by = 'LotArea', ascending=False)[:1] # last value to be deleted

df_train = df_train.drop(df_train[df_train['Id'] == 314].index)
df_train.sort_values(by = 'LotFrontage', ascending=False)[:1] # last value

df_train = df_train.drop(df_train[df_train['Id']== 935].index)
# Checking for normality

from scipy import stats

sns.distplot(df_train['SalePrice'], fit=stats.norm);# have to import norm in order to use

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt) #if no plot then a plot will not be created
# In case positive skewness we usually prefer log transformations

df_train['SalePrice'] = np.log(df_train['SalePrice'])
# Checking again after log transformation

sns.distplot(df_train['SalePrice'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
cols = df_train.select_dtypes(include = [np.number])

for i in cols.columns:

    if i !='Id':

        sns.distplot(df_train[i],fit = stats.norm)

        plt.xlabel(i)

        fig = plt.figure()

        stats.probplot(df_train[i], plot=plt)

        plt.show()
# log transformation for GrLivArea

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
# Checking again for GrLivArea after log transformation

sns.distplot(df_train['GrLivArea'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
# number of houses for which GarageArea is zero

sum(df_train['GarageArea']==0)
# log trnasformation for GarageArea

df_train['GarageArea'] = np.log(df_train['GarageArea'])
df_train['GarageArea'] = df_train['GarageArea'].replace(-np.inf, np.nan);

#imputing zero in place of NaN after log transformation

impute('GarageArea', 0)

# Againg checking fr GarageArea after the transformation

sns.distplot(df_train[df_train['GarageArea']>0]['GarageArea'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train[df_train['GarageArea']>0]['GarageArea'], plot = plt)
sum(df_train['TotalBsmtSF'] == 0)
# log trnasformation for TotalBsmtSF

df_train['TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
df_train['TotalBsmtSF'] = df_train.TotalBsmtSF.replace(-np.inf, np.nan)

# df_train.TotalBsmtSF.replace(np.inf, np.nan)

#imputing zero in place of NaN after log transformation

impute('TotalBsmtSF', 0)
# Againg checking fr TotalBsmtSF  after the transformation

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot = plt)
# log trnasformation for 1stFlrSF

df_train['1stFlrSF'] = np.log1p(df_train['1stFlrSF'])

# checking after log transformation

sns.distplot(df_train['1stFlrSF'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train['1stFlrSF'], plot= plt)
# LotArea

sns.distplot(df_train['LotArea'], fit = stats.norm)

fig = plt.figure()

stats.probplot(df_train['LotArea'], plot = plt);
df_train.LotArea.skew()
# Log transformation for LotArea

df_train['LotArea'] = np.log(df_train['LotArea'])

# Plot

sns.distplot(df_train['LotArea'], fit = stats.norm)

fig = plt.figure()

stats.probplot(df_train['LotArea'], plot = plt);
# LotFrontage

sns.distplot(df_train['LotFrontage'], fit = stats.norm)

fig = plt.figure()

stats.probplot(df_train['LotFrontage'], plot = plt);
# Log transformation for LotFrontage

df_train['LotFrontage'] = np.log(df_train['LotFrontage'])

# Plot

sns.distplot(df_train['LotFrontage'], fit = stats.norm)

fig = plt.figure()

stats.probplot(df_train['LotFrontage'], plot = plt);

# df_train.LotFrontage
# MSSubclass

sns.distplot(df_train['MSSubClass'], fit = stats.norm)

fig = plt.figure()

stats.probplot(df_train['MSSubClass'], plot=plt);
# Log transformation for MSSubclass

df_train['MSSubClass'] = np.log(df_train['MSSubClass'])

# Plot

sns.distplot(df_train['MSSubClass'], fit = stats.norm)

fig = plt.figure()

stats.probplot(df_train['MSSubClass'], plot = plt);
df_test[['MSSubClass', 'GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF',\

         'LotArea','LotFrontage']] = np.log(df_test[['MSSubClass', 'GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF',\

         'LotArea','LotFrontage']])
df_test.describe()
impute_test('LotFrontage',df_test['LotFrontage'].mean())
df_test['GarageArea'] = df_test['GarageArea'].replace(-np.inf, np.nan);

df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].replace(-np.inf, np.nan);

impute_test('GarageArea', df_test['GarageArea'].mean())

impute_test('TotalBsmtSF', df_test['TotalBsmtSF'].mean())

df_test.describe()
# train and test data

X_train = df_train.drop(['Id','SalePrice'], axis=1)

X_train['MSSubClass'] = np.log(X_train['MSSubClass'])

house_data = house_data.drop(house_data[house_data['Id'] == 1299].index)

house_data = house_data.drop(house_data[house_data['Id'] == 524].index)

house_data = house_data.drop(house_data[house_data['Id'] == 333].index)

house_data = house_data.drop(house_data[house_data['Id'] == 497].index)

house_data = house_data.drop(house_data[house_data['Id'] == 636].index)

y_train = df_train['SalePrice']

X_test = df_test.drop(['Id'], axis=1)

X_train = X_train.drop('CentralAir', axis=1)

X_test = X_test.drop('CentralAir', axis=1)
# get dummy variables

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)
X_test.columns
X_train.columns
X_train = X_train.drop('Electrical_Mix', axis=1)
from sklearn.linear_model import LinearRegression

logreg = LinearRegression()

logreg.fit(X_train, y_train)

#y_pred_lr = logreg.predict(X_test)

logreg.score(X_train, y_train)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge, RidgeCV, LassoCV, ElasticNet
# defining a function that returns rmse of cross_val error

def rmse(model):

    rmse_cv = np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5))

    return rmse_cv
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse(Ridge(alpha = a)).mean() for a in alphas] # mean cv_error for each alpha
cv_ridge = pd.Series(cv_ridge, index=alphas)

plt.plot(cv_ridge)

plt.title('Validation')

plt.xlabel('alphas')

plt.ylabel('rmse')
cv_ridge.min()
# Lasso (we will use builtin LassoCV to find best alpha)

model_lasso = LassoCV(alphas=[1,0.1,0.001,0.0005]).fit(X_train, y_train)
rmse(model_lasso).mean()
lasso_coef = pd.Series(model_lasso.coef_, index = X_train.columns)

imp_coef = pd.concat([lasso_coef.sort_values().head(10), lasso_coef.sort_values().tail(10)])
fig = plt.figure(figsize=(12,6))

imp_coef.plot(kind='barh', fontsize=12)

plt.show()
# Residual plot for lassoCV

preds = pd.DataFrame({"pred":model_lasso.predict(X_train), "true": y_train})

residual = preds['pred'] - preds['true']

fig = plt.figure(figsize=(6,6))

plt.scatter(preds['pred'],residual)

plt.show()
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import KFold

from scipy.stats import randint, uniform
xgb_tranin = xgb.DMatrix(X_train, label=y_train)

xgb_test = xgb.DMatrix(X_test)



xgb_para ={'max_depth':2, 'eta':0.1}

clf = xgb.cv(xgb_para,xgb_tranin, num_boost_round=500, early_stopping_rounds=100)
clf.loc[30:,['test-rmse-mean', 'train-rmse-mean']].plot(figsize=(8,6))
cv = KFold(n_splits=2, shuffle=True, random_state=123)

params_dist_grid = { 'max_depth': [5,10,15], 

                     'n_estimators':randint(1,1001),

                     'learning_rate':uniform(), # Gaussian Distribution

                     'subsample': uniform(), # Gaussian Distribution

                     'colsample_bytree': uniform(), # Gaussian Distribution

                     'reg_lambda':uniform(), # Gaussian Distribution

                     'reg_alpha':uniform(), # Gaussian Distribution

                     'max_delta_step': [0,1,2],

                     'gamma':uniform() # Gaussian Distribution

                   }
bst_grid = RandomizedSearchCV(estimator=xgb.XGBRegressor(), param_distributions=params_dist_grid, cv=cv)
bst_grid.fit(X_train, y_train)
bst_grid.grid_scores_
print ('Best accuracy obtained: {}'.format(bst_grid.best_score_))

print ('Parameters:')

for key, value in bst_grid.best_params_.items():

    print('\t{}:{}'.format(key,value))
model = xgb.XGBRegressor(n_estimators=100, max_depth=15, learning_rate=0.07,reg_alpha=0.25,colsample_bytree=0.129,\

                        max_delta_step=0, subsample=0.288, reg_lambda=0.45, gamma=0.47)

model.fit(X_train, y_train)
y_pred = np.exp(model.predict(X_test))

solution = pd.DataFrame({'Id':df_test.Id, 'SalePrice':y_pred}, columns = ['Id','SalePrice'])

solution.to_csv('House_Price.csv', index=False)