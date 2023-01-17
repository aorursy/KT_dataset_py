# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Importing the libraries for ploting

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing the datasets

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
#Lets check the dataset

df_train.shape, df_test.shape, df_sub.shape
df_train.info()
df_test.info()
#At first we check our target variable

sns.set_style("whitegrid")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

sns.distplot(df_train['SalePrice'],color="b")

sns.despine(trim=True, left=True)

plt.xlabel('SalePrice')

plt.ylabel('Frequency')

plt.title('Distribution of Saleprice')

print('Skewness: %f', df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())

plt.show()
#Log transformation  

df_train['SalePrice_Log'] = np.log(df_train['SalePrice'])

print('Skewness: %f', df_train['SalePrice_Log'].skew())

print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())

sns.set_style("whitegrid")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

sns.distplot(df_train['SalePrice_Log'], color ='blue')

plt.xlabel('SalePrice_Log')

plt.ylabel('Frequency')

plt.title('Distribution of Saleprice_Log')

plt.show()
#Lets drop the  SalePrice column

df_train.drop('SalePrice', axis =1, inplace = True)
fig = plt.figure(figsize=(20, 20))

ax1 = plt.subplot2grid((3,2),(0,0))

plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice_Log'], color=('yellowgreen'), alpha=0.5)

plt.axvline(x=4600, color='r', linestyle='-')

plt.title('Ground living Area- Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((3,2),(0,1))

plt.scatter(x=df_train['TotalBsmtSF'], y=df_train['SalePrice_Log'], color=('red'),alpha=0.5)

plt.axvline(x=5900, color='r', linestyle='-')

plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )
fig = plt.figure(figsize=(20,20))

ax1 = plt.subplot2grid((3,2),(0,0))

plt.scatter(x=df_train['1stFlrSF'], y=df_train['SalePrice_Log'], color=('deepskyblue'),alpha=0.5)

plt.axvline(x=4000, color='r', linestyle='-')

plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((3,2),(0,1))

plt.scatter(x=df_train['MasVnrArea'], y=df_train['SalePrice_Log'], color=('gold'),alpha=0.9)

plt.axvline(x=1500, color='r', linestyle='-')

plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold' )
fig = plt.figure(figsize=(20,20))

ax1 = plt.subplot2grid((3,2),(0,0))

plt.scatter(x=df_train['GarageArea'], y=df_train['SalePrice_Log'], color=('orchid'),alpha=0.5)

plt.axvline(x=1230, color='r', linestyle='-')

plt.title('Garage Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((3,2),(0,1))

plt.scatter(x=df_train['TotRmsAbvGrd'], y=df_train['SalePrice_Log'], color=('orange'),alpha=0.9)

plt.axvline(x=13, color='r', linestyle='-')

plt.title('TotRmsAbvGrd - Price scatter plot', fontsize=15, weight='bold' )
# Now remove outliers

df_train.drop(df_train[df_train['GrLivArea'] > 4600].index, inplace=True)

df_train.drop(df_train[df_train['MasVnrArea'] > 1500].index, inplace=True)

df_train.shape 
#Lets check the correlation and heat map

corr = df_train.corr()

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.figure(figsize = (20,16))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor='white')

plt.title('Correlation of df Features', y = 1.05, size=15)
#Lets look the correlation score

print (corr['SalePrice_Log'].sort_values(ascending=False), '\n')
## Scatter plotting for and Putting a regression line.

fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot2grid((3,2),(0,0))

sns.scatterplot(x=df_train['OverallQual'], y=df_train['SalePrice_Log'], color=('red'),alpha=0.5)

sns.regplot(x=df_train['OverallQual'], y=df_train['SalePrice_Log'], color='red')

plt.title('correlation of SalePrice_Log and OverallQual',weight='bold' )



ax1 = plt.subplot2grid((3,2),(0,1))

sns.scatterplot( x = df_train['GrLivArea'], y = df_train['SalePrice_Log'], color = 'red')

sns.regplot(x=df_train['GrLivArea'], y=df_train['SalePrice_Log'], color = 'deeppink')

plt.title('correlation of SalePrice_Log and GrLivArea',weight='bold' )
# Scatter plotting and putting regression line

fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot2grid((3,2),(0,0))

sns.scatterplot(x = df_train.GarageArea,y = df_train.SalePrice_Log , color= 'green')

sns.regplot(x=df_train.GarageArea, y=df_train.SalePrice_Log, color = 'green')

plt.title('corelation of SalePrice_Log and GarageArea',weight='bold' )



ax1 = plt.subplot2grid((3,2),(0,1))

sns.scatterplot(x = df_train. TotalBsmtSF,y = df_train.SalePrice_Log , color= 'orange')

sns.regplot(x=df_train.TotalBsmtSF, y=df_train.SalePrice_Log, color = 'orange')

plt.title('corelation of SalePrice_Log and TotalBsmtSF',weight='bold' )
#We can see almost same columns of both train and test set have missing values. So we join the datasets

all_data = df_train.append(df_test)

all_data.shape
#Lets clean the dataset. Note that we don't need 'ID' to predict the value of the house. Lets drop the columns

all_data.drop('Id', axis =1, inplace = True)
#Lets ckeck again the missing value

all_data.isnull().sum()
#We note the LotFrontage has almost 20% missing value, we replace the nan value with mean

all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].mean())
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')

all_data['Alley'] = all_data['Alley'].fillna('None')

all_data['Fence'] = all_data['Fence'].fillna('None')

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
for col in ('GarageType', 'GarageFinish', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')

for col in ('GarageArea', 'GarageCars', 'GarageYrBlt', 'GarageQual'):

    all_data[col] = all_data[col].fillna(0)
for feat in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[feat] = all_data[feat].fillna('None')

for feat in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF'):

    all_data[feat] = all_data[feat].fillna(0)
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)

all_data['Functional'] = all_data['Functional'].fillna('Typical')

all_data['MSSubClass'] = all_data['MSSubClass'].fillna('None') 
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities']).mode()[0]   
## Some of the non-numeric predictors are stored as numbers; convert them into strings 

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
#get numeric features

numeric_features = [f for f in all_data.columns if all_data[f].dtype != object]
#transform the numeric features using log(x + 1)

from scipy.stats import skew

skewed = all_data[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))

skewed = skewed[skewed > 0.75]

skewed = skewed.index

all_data[skewed] = np.log1p(all_data[skewed])
#add TotalSF

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#Log features

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

all_data = logs(all_data, log_features)
#Square features

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

all_data = squares(all_data, squared_features)
all_data = pd.get_dummies(all_data).reset_index(drop=True)

all_data.shape
# Remove any duplicated column names

all_data = all_data.loc[:,~all_data.columns.duplicated()]
train_new = all_data[all_data['SalePrice_Log'].notnull()]

test_new = all_data[all_data['SalePrice_Log'].isnull()]

test_new = test_new.drop(['SalePrice_Log'], axis = 1)
train_new.shape, test_new.shape
y_train = train_new['SalePrice_Log'].values
x = train_new.drop(['SalePrice_Log'], axis = 1)

x_train = x.iloc[:,:].values

x_test = test_new.iloc[:,:].values
from sklearn.preprocessing import RobustScaler

RS = RobustScaler()

x_train = RS.fit_transform(x_train)

x_test = RS.transform(x_test)
#Lets use the Lasso regression

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso

lasso = Lasso()
# list of alphas to tune

params = {'alpha': [0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.008, 0.009]}
# cross validation

grid_search = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = 10, 

                        return_train_score=True,

                        verbose = 1)            

grid_search.fit(x_train, y_train)
#checking the value of optimum number of parameters

print(grid_search.best_params_)

print(grid_search.best_score_)

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=1000]

cv_results
#Fitting Lasso Regression to the tranning set

lasso = Lasso(alpha = 0.001, max_iter = 1000)

lasso.fit(x_train, y_train)
#Lets check the accuracy 

accuracy_train = lasso.score(x_train, y_train)

print(accuracy_train)
#Xgboot regression

from xgboost import XGBRegressor

xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3000,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)



xgboost.fit(x_train,y_train)
accuracy_train = xgboost.score(x_train, y_train)

print(accuracy_train)
from lightgbm import LGBMRegressor

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

                       random_state=42)

lightgbm.fit(x_train, y_train)
accuracy_train = lightgbm.score(x_train, y_train)

print(accuracy_train)
#Gradient boost regression

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =42)

gbr.fit(x_train, y_train)
accuracy_train = gbr.score(x_train, y_train)

print(accuracy_train)
##Now predicting the test set result

y_pred = 0.25*lasso.predict(x_test)+0.25*gbr.predict(x_test)+0.25*lightgbm.predict(x_test)+0.25*xgboost.predict(x_test)
#convert back from logarithmic values to SalePrice

y_pred =np.expm1(y_pred) 
#Output

sub = pd.DataFrame()

sub['Id'] = df_test['Id']

sub['SalePrice'] = y_pred
# Fix outleir predictions

q1 = sub['SalePrice'].quantile(0.0045)

q2 = sub['SalePrice'].quantile(0.99)

sub['SalePrice'] = sub['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

sub['SalePrice'] = sub['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

# Scale predictions

sub['SalePrice'] *= 1.001619
sub.to_csv('submission.csv', index=False)