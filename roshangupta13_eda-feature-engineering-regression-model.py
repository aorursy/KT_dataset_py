import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns # for making plots with seaborn

import scipy as sp # Math functions

from scipy import stats

from scipy.stats import norm, skew, kurtosis

from scipy.special import boxcox1p

import matplotlib.pyplot as plt

import time



# scikit-learn modules

import sklearn

print("scikit-learn version: {}". format(sklearn.__version__))

# sklearn modules for preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# sklearn modules for ML model selection

from sklearn.model_selection import train_test_split  

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error



# Libraries for data modelling

from sklearn.linear_model import LassoCV, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor 

from sklearn.ensemble  import GradientBoostingRegressor



# Common sklearn Model Helpers

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler, RobustScaler



# Other ML algorithms

from mlxtend.regressor import StackingCVRegressor, StackingRegressor

print("StackingRegressor imported")
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
#import training and testing dataset

train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")

#Display shape of the dataset

print("training_data(shape) : {}".format(train_data.shape))

print("testing_data(shape) : {}".format(test_data.shape))
#Extract the ID of both csv

train_ID=train_data["Id"]

test_ID=test_data["Id"]

#drop redundant ID columns

train_data.drop('Id', axis=1, inplace=True)

test_data.drop('Id', axis=1, inplace=True)

#print their shape

print("training_ID(shape) : {}".format(train_ID.shape))

print("testing_ID(shape) : {}".format(test_ID.shape))
#Extract target variable for feature Engineering

SalePrice=train_data['SalePrice']

#drop SalePrice from dataframe

train_data.drop('SalePrice', axis=1, inplace=True)

print("SalePrice(shape): {}".format(SalePrice.shape))
#concat both the dataframes 

all_data=pd.concat([train_data,test_data], ignore_index="true", axis=0)

all_data.reset_index()

print("Shape of all data: {}".format(all_data.shape))
#print columns of dataframe

print(all_data.columns)

print()

print("Total no of columns : {}".format(len(all_data.columns)))
#lets explore our target varibale

#get the kurtosis of SalePrice

print("kurtosis: %f" % SalePrice.kurtosis())

#get the Skewness of SalePrice

print("Skewness: %f" % SalePrice.skew())

# calculate mean and standard deviation

(mu, sigma) = norm.fit(SalePrice)

print('Normal dist. (mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma))
#plotting diffrent plots to detect outliers

fig = plt.figure(figsize=(22,14))

ax1 = fig.add_subplot(331)

plt.title('original distribution')

plt.ylabel('Frequency')

sns.distplot(SalePrice, kde=True, fit=stats.lognorm, color="blue");

ax2 = fig.add_subplot(332)

stats.probplot(SalePrice, plot=plt);

ax3 = fig.add_subplot(333)

plt.title('boxplot')

sns.boxplot(SalePrice,orient="v")

fig.show()
#Function to count number of outliers

def count_outliers(target,state):

    # count number of outliers before and after transformation is applied

    if state=="before":

        Q1 = target.quantile(0.25)

        Q3 = target.quantile(0.75)

        IQR = Q3 - Q1

        print("Before transformation of SalePrice")

        print("Extreme outliers(Above 700000) : {} ".format([x for x in target if x > 700000]))

        print("IQR value: {}\nTotal outliers: {}".format(IQR, ((target < (Q1 - 1.5 * IQR)) | (target > (Q3 + 1.5 * IQR))).sum()))      

    if state=="after":

        Q1 = target.quantile(0.25)

        Q3 = target.quantile(0.75)

        IQR = Q3 - Q1

        print("After transformation of SalePrice")

        print("Extreme outliers(Above 700000) : {} ".format([x for x in target if x > 700000]))

        print("IQR value: {}\nTotal outliers: {}".format(IQR, ((target < (Q1 - 1.5 * IQR)) | (target > (Q3 + 1.5 * IQR))).sum()))
count_outliers(SalePrice,"before")
#Apply and check different transformation on SalePrice 

def target_transform(target,transformation=None):

    if(transformation is None):

        print("{} transformation ".format("Log"))

        #Log transformation for target variable

        #count no of zeores in series

        count_zero=sum(val == 0 for val in target)

        if(count_zero > 0):

            print("Transformation not applicable for zeroes values")

        else:

            SalePrice=np.log1p(target)

            return SalePrice

    #Sqrt transformation for target variable        

    if(transformation=="sqrt"):

        print("{} transformation ".format(transformation))

        SalePrice=np.sqrt(target)

        return SalePrice

    #boxcox1p transformation for target variable    

    if(transformation=="boxcox1p"):

        print("{} transformation ".format(transformation))

        SalePrice=boxcox1p(target , 0.2)

        return SalePrice
# since there is no zero values so Log transformation is suitable, eliminates outliers and decreases the range

SalePrice=target_transform(SalePrice)
#lets explore our target varibale log transformation

#get the kurtosis of SalePrice

print("kurtosis: %f" % SalePrice.kurtosis())

#get the Skewness of SalePrice

print("Skewness: %f" % SalePrice.skew())

# calculate mean and standard deviation

(mu, sigma) = norm.fit(SalePrice)

print('Normal dist. (mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma))
#After log transformation

count_outliers(SalePrice,"after")
#plotting diffrent plots log transformation

fig = plt.figure(figsize=(22,14))

ax1 = fig.add_subplot(331)

plt.title('Log distribution')

plt.ylabel('Frequency')

sns.distplot(SalePrice, kde=True, fit=stats.lognorm, color="blue");

ax2 = fig.add_subplot(332)

stats.probplot(SalePrice, plot=plt);

ax3 = fig.add_subplot(333)

plt.title('boxplot')

sns.boxplot(SalePrice,orient="v")

fig.show()
SalePrice.describe()
#count categorical and numerical columns

categorical_cols=[col for col in all_data.columns if all_data[col].dtype =='O']

numerical_cols=[col for col in all_data.columns if all_data[col].dtype !='O']

print("categorical_columns : \n")

print(categorical_cols)

print("\n")

print("numerical_columns :\n")

print(numerical_cols)
#lets find the missing value percentage

null_percentage = (all_data.isnull().sum()/len(all_data))*100

null_ratio = null_percentage.sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Values Ratio' :null_ratio})

print(missing_data.shape)

missing_data.head(10)
print("Number of columns has a missing data")

print(len(null_percentage.index) - len([x for x in null_percentage.index if null_percentage.loc[x]==0]))
#Handle missing data

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)    

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data = all_data.drop(['Utilities'], axis=1)

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')  

for col in ['Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    
null_percentage = (all_data.isnull().sum()/len(all_data))*100

null_ratio = null_percentage.sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Values Ratio' :null_ratio})

print("Number of columns has a missing data")

print(len(null_percentage.index) - len([x for x in null_percentage.index if null_percentage.loc[x]==0]))
#Converting some numerical variables that are really categorical type

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
cols_label = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold']

#Convert Features in label encoding

enc_label=LabelEncoder()

for columns in cols_label:

    all_data[columns] = enc_label.fit_transform(all_data[columns]) 
numerical_features=all_data.dtypes[all_data.dtypes != "object"].index

skew_percentage = all_data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skew_percentage})

skewness.head(10)
#log transformation for top 5 features with high skewness

skew_feat=skewness.iloc[:5]

for columns in skew_feat.index:

    all_data[columns] = boxcox1p(all_data[columns],0.2) 

 
# scale all numerical feature 

scale=StandardScaler()

numerical_features=all_data.dtypes[all_data.dtypes != "object"].index

all_data[numerical_features]=scale.fit_transform(all_data[numerical_features].values)
# get dummies for categorical variable

all_data=pd.get_dummies(all_data)
# split the data in training and prediction set

train_set=all_data[:len(train_ID)]

predict_set=all_data[len(test_ID)+1:]
#splits the dataset into training and testing set

X_train, X_test, Y_train, Y_test = train_test_split(train_set, SalePrice, test_size=0.20, random_state=42)
model_name=[]

r_sqaure_value=[]

mean_sqaure_value=[]

def calculate_metrics(model, predicted_value, test_value ):

    if model not in model_name:

        model_name.append(model)

        r_sqaure_value.append(r2_score(predicted_value,test_value))

        mean_sqaure_value.append(mean_squared_error(predicted_value,test_value))

#         print(model_name)

#         print(r_sqaure_value)

#         print(mean_sqaure_value)
# generalized ridge regression with builtin cross validation

params={'alphas':[0.001, 0.01 , 0.1 ,1.0],

 'cv': 5,

 'fit_intercept': True,

 'gcv_mode': None,

 'normalize': False,

 'scoring': None,

 'store_cv_values': False}



ridge_reg=RidgeCV(**params)

ridge_reg.fit(X_train,Y_train)

ridge_predicted=ridge_reg.predict(X_test)

calculate_metrics("LassoCv", ridge_predicted, Y_test)
# generalized lasso regression with builtin cross validation



params={'alphas':[0.1,0.001,0.003,0.0001],

 'copy_X': True,

 'cv': 5,

 'eps': 0.001,

 'fit_intercept': True,

 'max_iter': 2000,

 'n_alphas': 100,

 'n_jobs': None,

 'normalize': False,

 'positive': False,

 'precompute': 'auto',

 'random_state': None,

 'selection': 'cyclic',

 'tol': 0.0001,

 'verbose': False}



lasso_reg = LassoCV(**params)

lasso_reg.fit(X_train,Y_train)

lasso_predicted=lasso_reg.predict(X_test)

calculate_metrics("RidgeCV", lasso_predicted, Y_test)
# generalized ElasticNetCV with builtin cross validation

params={'alphas': [0.1,0.001, 0.02,1.0],

 'copy_X': True,

 'cv': 5,

 'eps': 0.001,

 'fit_intercept': True,

 'l1_ratio': [0.5,0.4],

 'max_iter': 3000,

 'n_alphas': 100,

 'n_jobs': None,

 'normalize': False,

 'positive': False,

 'precompute': 'auto',

 'random_state': None,

 'selection': 'cyclic',

 'tol': 0.0001,

 'verbose': 0}



Enet=ElasticNetCV(**params)

Enet.fit(X_train,Y_train)

Enet_predicted=Enet.predict(X_test)

calculate_metrics("ElasticNetCV", Enet_predicted, Y_test)
#Grid Search for SVR

start = time.time()

print("Grid Search Started : {}".format(start))

grid_ser_svr = GridSearchCV(

        estimator=SVR(),

        param_grid={'C': [0.1, 1, 0.2],

                    'epsilon': [0.0001, 0.001, 0.005],

                    'gamma': [0.0001, 0.001, 0.1 ],

                    'kernel': ["linear", "poly", "rbf"] 

                   },

        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

svr_grid= grid_ser_svr.fit(X_train,Y_train)

end = time.time()

print("Grid Search ended: {}".format(end))

print("Elapsed time for Grid search: {}".format(end-start))

print("Best parameters for grid search\n")

print(svr_grid.best_params_)

print("\n")

calculate_metrics("SVR", svr_grid.predict(X_test), Y_test)
#Implementation of Decision tree

params={'criterion': 'mse',

 'max_depth':  None,

 'max_features': None,

 'max_leaf_nodes': None,

 'min_impurity_decrease': 0.0,

 'min_impurity_split': None,

 'min_samples_leaf': 1,

 'min_samples_split': 2,

 'min_weight_fraction_leaf': 0.0,

 'presort': True,

 'random_state': None,

 'splitter': 'best'}

deci_tree_reg=DecisionTreeRegressor(**params)

deci_tree_reg.fit(X_train,Y_train)

decision_predicted=deci_tree_reg.predict(X_test)

calculate_metrics("Decision Tree", decision_predicted, Y_test)
#Grid Search for Random Forest Regressor

start = time.time()

print("Grid Search Started : {}".format(start))

grid_ser_rand = GridSearchCV(

        estimator=RandomForestRegressor(),

        param_grid={'max_depth': [60, 50, 40, 45],

                   'n_estimators': [5,6,7, 8],

                   'max_leaf_nodes': [2, 3, 4,5,6]},

        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

random_grid= grid_ser_rand.fit(X_train,Y_train)

end = time.time()

print("Grid Search ended: {}".format(end))

print("Time taken by  Grid search: {}".format(end-start))

print("Best parameters for grid search : \n")

print(random_grid.best_params_)

calculate_metrics("Random Forest", random_grid.predict(X_test), Y_test)
#Grid Search for gradient boosting Regressor

start = time.time()

print("Grid Search Started : {}".format(start))

grid_ser_boost = GridSearchCV(

        estimator=GradientBoostingRegressor(),

        param_grid={'max_depth': [3, 4, 6, 8],

                   'n_estimators': [100,120,80,90]},

        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

gra_grid= grid_ser_boost.fit(X_train,Y_train)

end = time.time()

print("Grid Search ended: {}".format(end))

print("Time taken by  Grid search: {}".format(end-start))

print("Best parameters for grid search\n")

print(gra_grid.best_params_)

calculate_metrics("GDBRT", gra_grid.predict(X_test), Y_test)
#plotting Accuracy of model

fig = plt.figure(figsize=(30,12))

ax1 = fig.add_subplot(221)

plt.title('Accuracy of Model')

plt.ylabel('R Square')

sns.set(style="whitegrid")

sns.barplot(x=model_name, y=r_sqaure_value)

fig.show()
#plotting Mean Square error of model

fig = plt.figure(figsize=(30,12))

ax1 = fig.add_subplot(221)

plt.title('Mean Squared Error of Model')

plt.ylabel('Mean Squared Error')

sns.set(style="whitegrid")

sns.barplot(x=model_name, y= mean_sqaure_value)

fig.show()
#Taking Average predicted price of all model

print("Predicting price of house")

final_price=(np.expm1(ridge_reg.predict(predict_set))

+np.expm1(svr_grid.predict(predict_set))

+np.expm1(lasso_reg.predict(predict_set))

+np.expm1(Enet.predict(predict_set))

+np.expm1(gra_grid.predict(predict_set))

+np.expm1(deci_tree_reg.predict(predict_set)))/6

final_price
submit = pd.DataFrame()

submit['Id'] = test_ID

submit['SalePrice'] = final_price

submit.to_csv('submission.csv',index=False)