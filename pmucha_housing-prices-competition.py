import pandas as pd # data processing, CSV file I/O

import numpy as np # linear algebra

%matplotlib inline

import matplotlib.pyplot as plt # charts

import seaborn as sns # charts

color = sns.color_palette()

sns.set_style('darkgrid')



from scipy import stats

from scipy.stats import norm, skew #for some statistics



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")



# let's see first 15 entries

data_train.head(15)
data_train.info()
data_train.describe()
# save the 'Id' column and drop the from dataset since it's unnecessary for  the prediction process.

data_train_ID = data_train['Id']

data_test_ID = data_test['Id']



data_train.drop("Id", axis = 1, inplace = True)

data_test.drop("Id", axis = 1, inplace = True)

print("Done")
correlation_matrix = data_train.corr()

plt.subplots(figsize=(16,16))

sns.heatmap(correlation_matrix, vmax=0.9)
# create an pandas-type-Index with 10 variables with highest correlation with SalePrice 

cols = correlation_matrix.nlargest(10,'SalePrice')['SalePrice'].index

cm = np.corrcoef(data_train[cols].values.T)

ht = sns.heatmap(cm,annot=True, annot_kws={'size': 10},fmt='.2f',

                 yticklabels=cols.values, xticklabels=cols.values

                ) #annot=show correlation value, annot_kws=size of annotation, fmt=round values to .2

plt.show()
#    Another way to plot:

#data = pd.concat([data_train.SalePrice,data_train.GrLivArea],axis = 1)

#data.plot.scatter(x='GrLivArea',y='SalePrice',ylim=(0,800000))



fig, ax = plt.subplots()

ax.scatter(x = data_train['GrLivArea'], y = data_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
data_train = data_train.drop(data_train[(data_train['GrLivArea'] > 4000) & 

                                        (data_train['SalePrice'] < 300000)].index)
data = pd.concat([data_train.SalePrice,data_train.GrLivArea],axis = 1)

data.plot.scatter(x='GrLivArea',y='SalePrice',ylim=(0,800000))

fig, ax = plt.subplots()

ax.scatter(x = data_train['TotalBsmtSF'], y = data_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)

plt.show()
sns.distplot(data_train['SalePrice'] , fit=norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(data_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(data_train['SalePrice'], plot=plt)

plt.show()
# log transformation usually works well with skewness

data_train['SalePrice'] = np.log1p(data_train['SalePrice'])
sns.distplot(data_train['SalePrice'] , fit=norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(data_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(data_train['SalePrice'], plot=plt)

plt.show()
print(data_train.shape)

print(data_test.shape)
ntrain = data_train.shape[0]

ntest = data_test.shape[0]

ytrain = data_train.SalePrice.values



all_data = pd.concat((data_train,data_test)).reset_index(drop = True)

all_data.drop(['SalePrice'], axis = 1, inplace = True)

print(all_data.shape)
total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending = False)

missing_data = pd.DataFrame(

                            {'Total' : total[total > 0],

                             'Missing Ratio': percent[percent > 0]

                            }

                            )

missing_data



# other versions:

#total = all_data.isnull().sum() 

#total = total.drop(total[total == 0].index).sort_values(ascending=False)

#percent = (total / len(all_data)) * 100

#missing_data = pd.concat([total,percent], axis = 1, keys=['Total','Missing ratio'])
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

# all_data.loc[all_data.LotFrontage.isna(),'LotFrontage'] = all_data[~all_data.LotFrontage.isna()].LotFrontage.mean()

# Replacing missing data with None

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')



# Replacing missing data with 0 (Since No garage = no cars in such garage.)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)



# For all these categorical basement-related features, NaN means that there is no basement.

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')



# missing values are likely zero for having no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

    

# NA most likely means no masonry veneer for these houses. 

# We can fill 0 for the area and None for the type.

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)



# 'RL' is by far the most common value. So we can fill in missing values with 'RL'

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



# data description says NA means typical

all_data["Functional"] = all_data["Functional"].fillna("Typ")



# It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])



# Only one NA value, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])



# Both Exterior 1 & 2 have only one missing value -> substitute in the most common string

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])



#Fill in again with most frequent which is "WD"

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])



# Na most likely means No building class. We can replace missing values with None

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")



# all records are "AllPub", except for one "NoSeWa" and 2 NA . 

# Since the house with 'NoSewa' is in the training set, 

# this feature won't help in predictive modelling. We can then safely remove it.

all_data = all_data.drop(['Utilities'], axis=1)



#Check remaining missing values if any 

total = all_data.isnull().sum()

total[total > 0]
#The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Overall condition

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
# check which columns are categorical

all_data.columns[all_data.dtypes == "object"]
# identify and create distinct values in categorical columns

from sklearn.preprocessing import LabelEncoder

for col in all_data.columns[all_data.dtypes == "object"]:

    all_data[col] = all_data[col].factorize()[0]

# after features engineering split back the dataset into train and test parts.

data_train = all_data.iloc[:ntrain]

data_test = all_data.iloc[ntrain:]
# split train data into two groups: training and validation set

# that we will use for tuning the model

from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(data_train, ytrain, test_size=0.3, random_state=42)
import xgboost as xgb

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_log_error

from bayes_opt import BayesianOptimization



# we will use hyperoptimastion to measure best values of parameters of XGBRegression model

# however, we train the model with n_jobs = 4, learning rate lowest possible 0.01, and seed=42

dtrain = xgb.DMatrix(data_train, label=ytrain)



def xgb_evaluate(max_depth, n_estimators,min_child_weight,subsample,

                 colsample_bytree,colsample_bylevel,reg_lambda):

    params = {

              'max_depth': int(max_depth),

              'n_estimators': n_estimators,

              'min_child_weight': min_child_weight,

              'n_jobs': 4,

              'subsample': subsample,

              'colsample_bytree': colsample_bytree,

              'learning_rate': 0.01,

              'reg_lambda': reg_lambda,

              'seed': 42,

              'silent':True

    }



    cv_result = xgb.cv(params, dtrain, num_boost_round = 500, nfold = 5, early_stopping_rounds=10)    

    

    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE

    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]



xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 5), 

                                             'n_estimators': (500,3500),

                                             'min_child_weight': (1.5,7.5),

                                             'subsample':(0.3,1.0),

                                             'colsample_bytree': (0.01,1.0),

                                             'colsample_bylevel': (0.01,0.2),

                                             'reg_lambda': (0.1,10.0)

                                             })



xgb_bo.maximize(init_points = 3, n_iter = 50, acq='ei')
# Bayesian Optimalisation gives us a hint about best paramteres

# after 53 loops best parameters we get are:

model = xgb.XGBRegressor(colsample_bylevel=0.25,colsample_bytree=0.6,max_depth=4,min_child_weight=4.8,n_estimators=3000,reg_lambda=0.1,subsample=0.8, learning_rate=0.01, n_jobs=4,seed=42)



# check the accuracy of cross valuation method

model.fit(train_x, train_y, early_stopping_rounds=5, eval_set=[(val_x,val_y)], verbose=False)

print("Valuated accuracy:", np.sqrt(mean_squared_log_error(model.predict(val_x), val_y)))

print("Trained accuracy:", np.sqrt(mean_squared_log_error(model.predict(train_x), train_y)))



# check the accuracy of the full train dataset

accuracy = cross_val_score(model, data_train, ytrain, cv=5, scoring="neg_mean_squared_log_error")

print("Accuracy of train data:", np.sqrt(-accuracy.mean()))

# training the model of the full dataset

model.fit(data_train, ytrain)
y_pred = model.predict(data_test)

Y_pred = np.expm1(y_pred)

Y_pred
submission = pd.DataFrame({

    "Id": data_test_ID, 

    "SalePrice": Y_pred 

})

submission.to_csv('submission.csv', index=False)