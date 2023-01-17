# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%reset -f



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.describe()

train.head()
train.tail()
# most correlated features

corrmat = train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']

sns.pairplot(train[cols], size = 2.5)

plt.show();
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
# Drop outliers on the basis of GrLivArea



train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train.shape
fig, ax = plt.subplots()

ax.scatter(x = train['OverallQual'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallQual', fontsize=13)

plt.show()
# Drop outliers on the basis of OverallQual



train = train.drop(train[(train['OverallQual']>=10) & (train['SalePrice']<300000)].index)
train.shape
# Exploring the sale price variable itself.



from scipy import stats

from scipy.stats import norm, skew #for some statistics



sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
train.SalePrice = np.log1p(train.SalePrice )

y = train.SalePrice

# Here we combine the train and test datasets so that we call get_dummies on them together. 

# This way get the same dimensions of feature vector.



train = train.drop("SalePrice", axis=1)



concat_train_test = pd.concat([train,test])

concat_train_test_na = (concat_train_test.isnull().sum() / len(concat_train_test)) * 100

concat_train_test_na = concat_train_test_na.drop(concat_train_test_na[concat_train_test_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :concat_train_test_na})

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=concat_train_test_na.index, y=concat_train_test_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
# Now handle all the missing data one by one



concat_train_test["PoolQC"] = concat_train_test["PoolQC"].fillna("None")

concat_train_test["MiscFeature"] = concat_train_test["MiscFeature"].fillna("None")

concat_train_test["Alley"] = concat_train_test["Alley"].fillna("None")

concat_train_test["Fence"] = concat_train_test["Fence"].fillna("None")

concat_train_test["FireplaceQu"] = concat_train_test["FireplaceQu"].fillna("None")



concat_train_test["LotFrontage"] = concat_train_test.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    concat_train_test[col] = concat_train_test[col].fillna('None')

    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    concat_train_test[col] = concat_train_test[col].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    concat_train_test[col] = concat_train_test[col].fillna(0)

    

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    concat_train_test[col] = concat_train_test[col].fillna('None')

    

concat_train_test["MasVnrType"] = concat_train_test["MasVnrType"].fillna("None")

concat_train_test["MasVnrArea"] = concat_train_test["MasVnrArea"].fillna(0)

concat_train_test['MSZoning'] = concat_train_test['MSZoning'].fillna(concat_train_test['MSZoning'].mode()[0])



concat_train_test = concat_train_test.drop(['Utilities'], axis=1)



# Smaller values

concat_train_test["Functional"] = concat_train_test["Functional"].fillna("Typ")

concat_train_test['Electrical'] = concat_train_test['Electrical'].fillna(concat_train_test['Electrical'].mode()[0])

concat_train_test['KitchenQual'] = concat_train_test['KitchenQual'].fillna(concat_train_test['KitchenQual'].mode()[0])

concat_train_test['Exterior1st'] = concat_train_test['Exterior1st'].fillna(concat_train_test['Exterior1st'].mode()[0])

concat_train_test['Exterior2nd'] = concat_train_test['Exterior2nd'].fillna(concat_train_test['Exterior2nd'].mode()[0])

concat_train_test['SaleType'] = concat_train_test['SaleType'].fillna(concat_train_test['SaleType'].mode()[0])

concat_train_test['MSSubClass'] = concat_train_test['MSSubClass'].fillna("None")

concat_train_test.shape
# Differentiate numerical features (minus the target) and categorical features

categorical_features = concat_train_test.select_dtypes(include=['object']).columns

categorical_features
numerical_features = concat_train_test.select_dtypes(exclude = ["object"]).columns
# Differentiate numerical features (minus the target) and categorical features

categorical_features = concat_train_test.select_dtypes(include = ["object"]).columns

numerical_features = concat_train_test.select_dtypes(exclude = ["object"]).columns

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

train_test_num = concat_train_test[numerical_features]

train_test_cat = concat_train_test[categorical_features]
# Handle remaining missing values for numerical features by using median as replacement

print("NAs for numerical features in train : " + str(train_test_num.isnull().values.sum()))

train_test_num = train_test_num.fillna(train_test_num.median())

print("Remaining NAs for numerical features in train : " + str(train_test_num.isnull().values.sum()))
train_test_cat = pd.get_dummies(train_test_cat)

train_test_cat.shape
# The actual modeling code

import pandas as pd

import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer

import matplotlib.pyplot as plt

import seaborn as sns
final_concat_train_test = pd.concat([train_test_cat,train_test_num],axis=1)

final_concat_train_test.shape
# Now we need to split our original test and train data again.



# First we sort by id.

final_concat_train_test.sort_values(by='Id', inplace=True)





split_test = final_concat_train_test[final_concat_train_test['Id'] > 1460]

split_train = final_concat_train_test[final_concat_train_test['Id'] <= 1460]
split_test.shape
split_train.shape
#split the data to train the model 

X_train,X_test,y_train,y_test = train_test_split(split_train,y,test_size = 0.3,random_state= 0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
X_train.head(3)
n_folds = 5

from sklearn import metrics

from sklearn.metrics import make_scorer

from sklearn.model_selection import KFold



def report_metrics(y_truth, y_predicted):

    print("MSE")

    print(metrics.mean_squared_error(y_truth, y_predicted))

    print("RMSE")

    print(np.sqrt(metrics.mean_squared_error(y_truth, y_predicted)))

# scorer = make_scorer(mean_squared_error,greater_is_better = False)

# def rmse_CV_train(model):

#     kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(split_train.values)

#     rmse = np.sqrt(-cross_val_score(model,X_train,y_train,scoring ="neg_mean_squared_error",cv=kf))

#     return (rmse)

# def rmse_CV_test(model):

#     kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(split_train.values)

#     rmse = np.sqrt(-cross_val_score(model,X_test,y_test,scoring ="neg_mean_squared_error",cv=kf))

#     return (rmse)
lr = LinearRegression()

lr.fit(X_train,y_train)

test_pre = lr.predict(X_test)

train_pre = lr.predict(X_train)

report_metrics(y_test, test_pre)
#plot between predicted values and residuals

plt.scatter(train_pre, train_pre - y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre,test_pre - y_test, c = "green",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
# Plot predictions - Real values

plt.scatter(train_pre, y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre, y_test, c = "black",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()
final_test_pre = lr.predict(split_test)
final_test_pre.shape
final_test_pre
final_test_pre = np.expm1(final_test_pre)
final_test_pre
output = pd.DataFrame({'Id': test.Id, 'SalePrice': final_test_pre})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")