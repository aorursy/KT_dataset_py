import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import cufflinks as cf

import warnings

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LassoCV

from yellowbrick.regressor import AlphaSelection

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

%matplotlib inline

sns.set_style('whitegrid')

warnings.filterwarnings("ignore")
#file directory

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#import data train

data_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data_train.head()
#check data train total rows and columns

print('Data Train rows:', data_train.shape[0])

print('Data Train columns:', data_train.shape[1])
#check data train columns and its types

print('Data Train columns:\n')

print(data_train.dtypes)
#import data test

data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

data_test.head()
#check data test rows and columns

print('Data Test rows:', data_test.shape[0])

print('Data Test columns:', data_test.shape[1])
data_train['SalePrice'].describe()
#sale price histogram

cf.go_offline()

data_train['SalePrice'].iplot(kind='hist',bins=30,color='red')
#correlation matrix heatmap

corrmat = data_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, cmap='coolwarm', vmax=.8, square=True);
#correlation matrix numbers

data_train.corr().style.background_gradient(cmap='coolwarm').set_precision(4)
#histogram and scatter plot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(data_train[cols], height = 2.5)

plt.show();
#missing data train heatmap

sns.heatmap(data_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#missing data test heatmap

sns.heatmap(data_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(data_train.shape))

print("The test data size before dropping Id feature is : {} ".format(data_test.shape))



#Save the 'Id' column

data_train_ID = data_train['Id']

data_test_ID = data_test['Id']



#Now drop the 'Id' column since it's unnecessary for the prediction process.

new_data_train = data_train.copy()

new_data_test = data_test.copy()

new_data_train.drop("Id", axis = 1, inplace = True)

new_data_test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(new_data_train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(new_data_test.shape))
price = new_data_train['SalePrice']
#combine data train and data test

ntrain = new_data_train.shape[0]

ntest = new_data_test.shape[0]

all_data = pd.concat((new_data_train, new_data_test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
#check ratio of null values

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
all_data[['GarageType','GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual','GarageCond']].head(len(all_data[['GarageType','GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual','GarageCond']]))
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
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
sns.heatmap(all_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#create dummy variables

all_data = pd.get_dummies(all_data)

print(all_data.shape)
clean_data_train = all_data[:ntrain]

clean_data_test = all_data[ntrain:]
#remove outliers

z = np.abs(stats.zscore(clean_data_train.select_dtypes(['int64','float64'])))

no_outliers_data_train = clean_data_train[(z < 3).all(axis=1)]

no_outliers_price = price[(z < 3).all(axis=1)]



print(no_outliers_data_train.shape)

print(no_outliers_price.shape)
#split data train

X_train, X_test, y_train, y_test = train_test_split(no_outliers_data_train, no_outliers_price, test_size = 0.1)
#Lasso regression model fitting



# Create a list of alphas to cross-validate against

alphas = np.logspace(-10, 1, 100)



# Instantiate the linear model and visualizer

model_lasso = LassoCV(alphas=alphas)

visualizer = AlphaSelection(model_lasso)

visualizer.fit(X_train, y_train)

visualizer.show()
#Lasso regression's score

print('Score:', model_lasso.score(X_train,y_train))



#RMSE

pred_in_train_lasso = model_lasso.predict(X_train)

pred_out_train_lasso = model_lasso.predict(X_test)

rmse_in_train_lasso = mean_squared_error(y_train, pred_in_train_lasso, squared=False)

rmse_out_train_lasso = mean_squared_error(y_test, pred_out_train_lasso, squared=False)



print('RMSE in Train:', rmse_in_train_lasso)

print('RMSE outside Train:', rmse_out_train_lasso)
#XGBoost regression model fitting

model_xgboost = XGBRegressor(random_state=0)

model_xgboost.fit(X_train,y_train)
#XGBoost regression's score

print('Score:', model_xgboost.score(X_train,y_train))



#RMSE

pred_in_train_xgboost = model_xgboost.predict(X_train)

pred_out_train_xgboost = model_xgboost.predict(X_test)

rmse_in_train_xgboost = mean_squared_error(y_train, pred_in_train_xgboost, squared=False)

rmse_out_train_xgboost = mean_squared_error(y_test, pred_out_train_xgboost, squared=False)



print('RMSE in Train:', rmse_in_train_xgboost)

print('RMSE outside Train:', rmse_out_train_xgboost)
#create all lasso models

from sklearn.linear_model import Lasso



all_lasso_models = []



for i in alphas:

    all_lasso_models.append(Lasso(i).fit(X_train,y_train))
#RMSE

pred_in_train_lasso_combination = 0

pred_out_train_lasso_combination = 0

n = len(all_lasso_models)



for i in all_lasso_models:

    pred_in_train_lasso_combination += 1/n*i.predict(X_train)

    pred_out_train_lasso_combination += 1/n*i.predict(X_test)



rmse_in_train_lasso_combination = mean_squared_error(y_train, pred_in_train_lasso_combination, squared=False)

rmse_out_train_lasso_combination = mean_squared_error(y_test, pred_out_train_lasso_combination, squared=False)



print('RMSE in Train:', rmse_in_train_lasso_combination)

print('RMSE outside Train:', rmse_out_train_lasso_combination)
#predict the lasso combination results

def meta_classifiers_transformation(independent_variables):

    all_lasso_combination_results = []



    for i in all_lasso_models:

        all_lasso_combination_results.append(i.predict(independent_variables))



    all_lasso_combination_results = pd.DataFrame(all_lasso_combination_results).T

    

    return(all_lasso_combination_results)
#create the model

model_lasso_xgboost = XGBRegressor(random_state=0)

model_lasso_xgboost.fit(meta_classifiers_transformation(X_train),y_train)
#Meta-Classifier's score

print('Score:', model_lasso_xgboost.score(meta_classifiers_transformation(X_train),y_train))



#RMSE

pred_in_train_lasso_xgboost = model_lasso_xgboost.predict(meta_classifiers_transformation(X_train))

pred_out_train_lasso_xgboost = model_lasso_xgboost.predict(meta_classifiers_transformation(X_test))

rmse_in_train_lasso_xgboost = mean_squared_error(y_train, pred_in_train_lasso_xgboost, squared=False)

rmse_out_train_lasso_xgboost = mean_squared_error(y_test, pred_out_train_lasso_xgboost, squared=False)



print('RMSE in Train:', rmse_in_train_lasso_xgboost)

print('RMSE outside Train:', rmse_out_train_lasso_xgboost)
#prediction of Lasso regression

final_pred = 0



for i in all_lasso_models:

    final_pred += 1/n*i.predict(clean_data_test)



#create data frame

df = {

    'Id' : data_test_ID,

    'SalePrice' : final_pred

}



df = pd.DataFrame(df)



df.to_csv('Output.csv', index = False)



df