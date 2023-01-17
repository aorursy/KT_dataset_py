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
!pip install ppscore

### https://github.com/8080labs/ppscore - An alternative/improvement to correlation matrix
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import ppscore as pps



#filter out warnings

import warnings 

warnings.filterwarnings('ignore')



from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.columns
train.head()
train.shape
train.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False, cmap='plasma')
train.isnull().sum().sort_values(ascending=False)[0:19]
test.isnull().sum().sort_values(ascending=False)[0:33]
train['SalePrice'].describe()
#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])



#Check the new distribution 

sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
#Correlation map to see how features are correlated with SalePrice

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
print(cols)
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show();
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)

all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
final_all_data = all_data
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape

str_cols  = all_data.select_dtypes(include = 'object').columns

for c in str_cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



print('Shape all_data: {}'.format(all_data.shape))
final_data = all_data
final_data.columns
corre = pps.matrix(final_data[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',

       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 'YearRemodAdd']])
corre.columns
corre1 = corre.pivot(index='x', columns='y', values='ppscore')
corre1.columns
plt.figure(figsize=(12,10))

sns.heatmap(corre1,annot=True,fmt=".02f",square = False)
#train_y = pd.concat([y_train,test['SalePrice']],axis=0)
train_x,train_y = all_data[:1460],y_train
train_x.shape
train_y.shape
train_x.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



#splitting the dataset as training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(train_x,train_y)

y_train = pd.DataFrame(y_train)

y_test = pd.DataFrame(y_test)
#building the model

linreg = LinearRegression()

linreg.fit(X_train, y_train)



#Accuracy

print("R-Squared Value for Training Set: {:.3f}".format(linreg.score(X_train, y_train)))

print("R-Squared Value for Test Set: {:.3f}".format(linreg.score(X_test, y_test)))
from sklearn.linear_model import Ridge



ridge = Ridge()

ridge.fit(X_train, y_train)



print('R-squared score (training): {:.3f}'.format(ridge.score(X_train, y_train)))

print('R-squared score (test): {:.3f}'.format(ridge.score(X_test, y_test)))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



ridge = Ridge(alpha=20)

ridge.fit(X_train_scaled, y_train)



print('R-squared score (training): {:.3f}'.format(ridge.score(X_train_scaled, y_train)))

print('R-squared score (test): {:.3f}'.format(ridge.score(X_test_scaled, y_test)))
from sklearn.linear_model import Lasso



lasso = Lasso(max_iter = 10000)

lasso.fit(X_train, y_train)



print('R-squared score (training): {:.3f}'.format(lasso.score(X_train, y_train)))

print('R-squared score (test): {:.3f}'.format(lasso.score(X_test, y_test)))
lasso = Lasso(alpha=100, max_iter = 10000)

lasso.fit(X_train, y_train)



print('R-squared score (training): {:.3f}'.format(lasso.score(X_train, y_train)))

print('R-squared score (test): {:.3f}'.format(lasso.score(X_test, y_test)))
from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV



n_estimators = [100, 500, 900]

depth = [3,5,10,15]

min_split=[2,3,4]

min_leaf=[2,3,4]

bootstrap = ['True', 'False']

verbose = [0]



hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':depth,

    #'criterion':criterion,

    'bootstrap':bootstrap,

    'verbose':verbose,

    'min_samples_split':min_split,

    'min_samples_leaf':min_leaf

    }



random_cv = RandomizedSearchCV(estimator=regressor,

                               param_distributions=hyperparameter_grid,

                               cv=5, 

                               scoring = 'neg_mean_absolute_error',

                               n_jobs = -1, 

                               return_train_score = True,

                               random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
regressor = RandomForestRegressor(bootstrap='False', max_depth=15, min_samples_leaf=3,

                      min_samples_split=4, n_estimators=500, verbose=0)

regressor.fit(X_train,y_train)
regressor.fit(X_train,y_train)

print('R-squared score (training): {:.3f}'.format(regressor.score(X_train, y_train)))

regressor.fit(X_test,y_test)

print('R-squared score (test): {:.3f}'.format(regressor.score(X_test, y_test)))
import xgboost
regressor=xgboost.XGBRegressor()

n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]

base_score=[0.25,0.5,0.75,1]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = -1,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
regressor = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.1, max_delta_step=0, max_depth=2,

             min_child_weight=1, monotone_constraints='()',

             n_estimators=900, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method='exact', validate_parameters=1, verbosity=None)
regressor.fit(X_train,y_train)

print('R-squared score (training): {:.3f}'.format(regressor.score(X_train, y_train)))

regressor.fit(X_test,y_test)

print('R-squared score (test): {:.3f}'.format(regressor.score(X_test, y_test)))
Test_X = all_data[1460:]

y_pred = regressor.predict(Test_X)
pred=pd.DataFrame(y_pred)

samp = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub = pd.concat([samp['Id'],pred], axis=1)

sub.columns=['Id','SalePrice']
sub
sub.to_csv('submission.csv',index=False)