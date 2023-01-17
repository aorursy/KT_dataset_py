import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

%matplotlib inline
#pd.set_option('display.max_rows', 500)

#pd.set_option('display.max_columns', 500)

# Read files

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print(test.shape)

test.head()
print(train.shape)

train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
#missing data percent plot

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=('Total', 'Percent'))

missing_data.head(20)
#missing data percent plot

total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=('Total', 'Percent'))

missing_data.head(34)
train['SalePrice'].describe()

sns.distplot(train['SalePrice']);

#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
from scipy import stats

from scipy.stats import norm, skew #for some statistics



# Plot histogram and probability

fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)





sns.distplot(train['SalePrice'] , fit=norm);

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1,2,2)

res = stats.probplot(train['SalePrice'], plot=plt)

plt.suptitle('Before transformation')





# Apply transformation

train.SalePrice = np.log1p(train.SalePrice )







# Plot histogram and probability after transformation

fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train['SalePrice'] , fit=norm);

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1,2,2)

res = stats.probplot(train['SalePrice'], plot=plt)

plt.suptitle('After transformation')
# train.SalePrice = np.log1p(train.SalePrice )



y_train = train.SalePrice.values   # type(train.SalePrice.values) = numpy.ndarray



y_train_orig = train.SalePrice  #y_train_orig is original sales Price
train.SalePrice.values
train.SalePrice
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' columns since they are unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
train.head(1)
print(train.shape)

print(test.shape)
# Dropping "SalePrice" before concating Test & Train so that we don't get NAN value after concat

#train.drop("SalePrice", axis = 1, inplace = True)
print(train.shape)
# data_features is our combined feature after Conacating

# It doesn't involve SalePrice Column

data_features = pd.concat((train, test)).reset_index(drop=True)

print(data_features.shape)
data_features.head()
data_features_na = data_features.isnull().sum()

data_features_na = data_features_na[data_features_na>0]

data_features_na.sort_values(ascending=False)
#missing data percent plot

total = data_features.isnull().sum().sort_values(ascending=False)

percent = (data_features.isnull().sum()/data_features.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=('Total', 'Percent'))

missing_data.head(35)
# missing value more than 80% fill then with 'None' means that feature is no available

for col in ['PoolQC','MiscFeature','Alley','Fence']:

    data_features[col] = data_features[col].fillna('None')

  
# Filling GarageX Variables

for col in [ 'GarageCond','GarageQual','GarageFinish','GarageType']:

    data_features[col] = data_features[col].fillna('None')
# Filling BsmtX Variables

for col in ['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType2','BsmtFinType1']:

    data_features[col] = data_features[col].fillna('None')

    
# FireplaceQu: Fireplace quality  NA	No Fireplace

data_features['FireplaceQu'] = data_features['FireplaceQu'].fillna('None')
sns.heatmap(data_features.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
# data description says NA means typical

data_features['Functional'] = data_features['Functional'].fillna('Typ')
# Replacing missing data with 0 (Since No garage = no cars in such garage.)

for col in ('BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF'):

    data_features[col] = data_features[col].fillna(0)
common_vars = ['Exterior1st','Exterior2nd','SaleType','KitchenQual']

for var in common_vars:

    data_features[var] = data_features[var].fillna(data_features[var].mode()[0])
data_features['MSZoning'] = data_features['MSZoning'].fillna(data_features['MSZoning'].mode())

data_features['Electrical'] = data_features['Electrical'].fillna(data_features['Electrical'].mode())
data_features['LotFrontage'] = data_features['LotFrontage'].fillna(data_features['LotFrontage'].mean())





data_features['MasVnrType'] = data_features['MasVnrType'].fillna(data_features['MasVnrType'].mode())

data_features['MasVnrType'] = data_features['MasVnrType'].fillna(data_features['MasVnrType'].mode())
data_features['GarageCars'] = data_features['GarageCars'].fillna(int(0))

data_features['GarageArea'] = data_features['GarageArea'].fillna(int(0))

data_features['GarageYrBlt'] = data_features['GarageYrBlt'].fillna(int(0))
data_features['Electrical'] = data_features['Electrical'].fillna(value='SBrkr')

data_features['Utilities'] = data_features['Utilities'].fillna(value='AllPub')

data_features['MSZoning'] = data_features['MSZoning'].fillna(value='RL')

data_features['MasVnrType'] = data_features['MasVnrType'].fillna(value='None')

data_features['MasVnrArea'] = data_features['MasVnrArea'].fillna(int(0))


sns.heatmap(data_features.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
# Differentiate numerical features (minus the target) and categorical features



categorical_features = data_features.select_dtypes(include=['object']).columns

print(categorical_features)

      

numerical_features = data_features.select_dtypes(exclude = ["object"]).columns

print(numerical_features)



print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

feat_num = data_features[numerical_features]

feat_cat = data_features[categorical_features]

feat_num.head()
feat_cat.head()
# Plot skew value for each numerical value

from scipy.stats import skew 

skewness = feat_num.apply(lambda x: skew(x))

skewness.sort_values(ascending=False)
skewness = skewness[abs(skewness) > 0.5]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

print("Mean skewnees: {}".format(np.mean(skewness)))
skewness.index
from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    feat_num[feat] = boxcox1p(feat_num[feat], lam)

    data_features[feat] = boxcox1p(data_features[feat], lam)

    

    

from scipy.stats import skew 

skewness.sort_values(ascending=False)
skewness = feat_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]



print("There are {} skewed numerical features after Box Cox transform".format(skewness.shape[0]))

print("Mean skewnees: {}".format(np.mean(skewness)))

skewness.sort_values(ascending=False)
len(y_train)
print(train.shape)

print(test.shape)
train = data_features.iloc[:len(y_train), :]

test = data_features.iloc[len(y_train):, :]

print(['Train data shape: ',train.shape,'Prediction on (Sales price) shape: ', y_train.shape,'Test shape: ', test.shape])


pd.set_option('display.max_columns', 1000)  # or 1000

pd.set_option('display.max_rows', 1000)

train.head(1)
test.head(1)
cols = list(train.columns.values) #Make a list of all of the columns in the df

cols.pop(cols.index('SalePrice')) #Remove b from list



train = train[cols+['SalePrice']] #Create new dataframe with columns in the order you want
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True,cmap="RdYlGn")

bottom,top = ax.get_ylim()

ax.set_ylim(bottom + 0.5,top - 0.5);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(15, 9))

hm = sns.heatmap(cm,cmap="RdYlGn", cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

bottom,top = ax.get_ylim()

ax.set_ylim(bottom + 0.5,top - 0.5);
train.drop('GarageArea', axis = 1, inplace = True)
train.drop('TotRmsAbvGrd', axis = 1, inplace = True)
data_features = pd.concat((train, test)).reset_index(drop=True)

print(data_features.shape)
data_features = data_features.drop("SalePrice", axis = 1)


final_features = pd.get_dummies(data_features)


final_features.shape
#There's a one line solution to the problem. This applies if some column names are duplicated and you wish to remove them:



#df = df.loc[:,~df.columns.duplicated()]




print(final_features.shape)

X_train = final_features.iloc[:len(y_train), :]

X_test = final_features.iloc[len(y_train):, :]

#.shape, y_train.shape, X_test.shape





print(X_train.shape,y_train.shape,X_test.shape)
X_train.shape
X_test= X_test.loc[:,~X_test.columns.duplicated()]
X_test.shape
y_train
import xgboost

classifier=xgboost.XGBRegressor()

import xgboost

regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]




n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
# Set up the random search with 4-fold cross validation

from sklearn.model_selection import RandomizedSearchCV

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.1, max_delta_step=0,

             max_depth=2, min_child_weight=1, missing=None, n_estimators=900,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)
regressor.fit(X_train,y_train)
X_test.shape
y_pred=regressor.predict(X_test)

y_pred
y_pred_aftr  = np.expm1(y_pred)
y_pred_aftr 
##Create Sample Submission file and Submit using ANN

pred=pd.DataFrame(y_pred_aftr)

sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission10122.csv',index=False)
pred.shape
sub_df=pd.read_csv('sample_submission10122.csv')

sub_df.head()