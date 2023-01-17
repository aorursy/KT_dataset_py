import numpy as np # linear algebra

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import xgboost as xgb

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from scipy.stats import skew

from sklearn.model_selection import cross_val_score, train_test_split

from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LassoCV



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import linear_model

from sklearn.ensemble import VotingClassifier,GradientBoostingRegressor



import warnings  

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

print("Shape of the train data:",train_data.shape)

print("Shape of the test data:",test_data.shape)
train_data.head()
#import pandas_profiling

#report=pandas_profiling.ProfileReport(train_data,check_correlation =True);

#report.to_file(outputfile="eda_report.html")

#report
#Check if the dataset is shuffled

plt.figure()

plt.plot(train_data.SalePrice,'.')

plt.title('SalesPrices within the train dataset')

plt.xlabel('Row Index')

plt.ylabel('SalePrice')

plt.show()
plt.figure()

sns.jointplot(x='OverallQual', y='SalePrice',data= train_data)

plt.show()
plt.figure(figsize=(50,25))

ax=sns.boxplot(x='YearBuilt',y='SalePrice',data=train_data)

plt.show()
#concatenate both train and test data

test_idx=test_data['Id']

all_data = pd.concat((train_data, test_data), sort=False).reset_index(drop=True)



#"SalePrice" is the target value. We don't include it in data. We don't want "id" affecting our model. Hence remove it.

all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data = all_data.drop(["Id"], axis=1)
all_nans = all_data.isnull().sum()

all_nans = all_nans[all_nans>0]

all_nans.sort_values(ascending=False)
#Imput missing values



all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')



all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].mode()[0])

all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data["KitchenQual"].mode()[0])

all_data["Functional"] = all_data["Functional"].fillna(all_data["Functional"].mode()[0])

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

for col in ('GarageType','GarageFinish','GarageQual','GarageCond'):

    all_data[col] = all_data[col].fillna("None")

for col in ('GarageYrBlt','GarageCars','GarageArea'):

    all_data[col] = all_data[col].fillna(0)

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].mode()[0])



#Now check again if there are anymore columns with missing values.

all_nans = all_data.isnull().sum()

all_nans = all_nans[all_nans>0]

all_nans.sort_values(ascending=False)



len(all_nans)
plt.figure()

sns.distplot(train_data['SalePrice'])

plt.title("Sale Price distribution skewed by {}".format(train_data.SalePrice.skew()))

plt.show()
train_data["SalePrice"] = np.log1p(train_data["SalePrice"])

plt.figure()

sns.distplot(train_data['SalePrice'])

plt.title("Sale Price distribution skewed by {}".format(train_data.SalePrice.skew()))

plt.show()
numerical_features = all_data.select_dtypes(exclude = ["object"]).columns

print("Number of numerical features: " + str(len(numerical_features)))



#log transform numerical features

skewness = all_data[numerical_features].apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.7]

skewed_features = skewness.index

all_data[skewed_features] = np.log1p(all_data[skewed_features])
categorical_features = all_data.select_dtypes(include = ["object"]).columns

print("Number of categorical features:" + str(len(categorical_features)))



#getdummies for categorical features

#Create a dataFrame with dummy categorical values

dummy_all_data = pd.get_dummies(all_data[categorical_features])

#Remove categorical features from original data, which leaves original data with only numerical featues

all_data.drop(categorical_features, axis=1, inplace=True)

#Concatenate the numerical features in original data and categorical features with dummies

all_data = pd.concat([all_data, dummy_all_data], axis=1)

#print(all_data.shape)
#Separate training and given test data

X = all_data[:train_data.shape[0]]

test_data = all_data[train_data.shape[0]:]

y = train_data["SalePrice"]

print(X.shape)

print(y.shape)
xgb1 = xgb.XGBRegressor(n_estimators=1000, max_depth=7,min_child_weight=1.5,reg_alpha=0.75,reg_lambda=0.45,learning_rate=0.07,subsample=0.95)

xgb1.fit(X, y)
xgb1_predicted_prices = np.expm1(xgb1.predict(test_data))

print(xgb1_predicted_prices)
my_submission = pd.DataFrame({'Id': test_idx, 'SalePrice': xgb1_predicted_prices})

my_submission.to_csv('submission1.csv', index=False)
xgb2 = xgb.XGBRegressor(n_estimators=1000, max_depth=3, min_child_weight=1.5,reg_alpha=0.75,reg_lambda=0.45, learning_rate=0.01, subsample=0.95)

xgb2.fit(X, y)
params={'n_estimators': 1000,'learning_rate': 0.01,'max_depth': 4}

gbm=GradientBoostingRegressor(**params)

gbm.fit(X, y)
lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X, y)

lasso_rmse=np.sqrt(-cross_val_score(lasso, X, y, scoring="neg_mean_squared_error", cv = 5)).mean()  

print('Lasso error', lasso_rmse)
#Make predictions on the test data

xgb2_predicted_prices = np.expm1(xgb2.predict(test_data))

gbm_predicted_prices =np.expm1(gbm.predict(test_data))

lasso_predicted_prices = np.expm1(lasso.predict(test_data))

predicted_prices = 0.50 * lasso_predicted_prices+0.25 * gbm_predicted_prices + 0.25 * xgb2_predicted_prices

print(predicted_prices)
#my_submission = pd.DataFrame({'Id': test_idx, 'SalePrice': predicted_prices})

#my_submission.to_csv('submission2.csv', index=False)