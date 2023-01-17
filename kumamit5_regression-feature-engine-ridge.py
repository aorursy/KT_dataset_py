!pip install feature_engine
#Import the libraries 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from feature_engine.missing_data_imputers import RandomSampleImputer,CategoricalVariableImputer

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder,OneHotCategoricalEncoder

from feature_engine.variable_transformers import PowerTransformer, YeoJohnsonTransformer

from sklearn.linear_model import Lasso,Ridge,LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost.sklearn import XGBRegressor

from sklearn.metrics import r2_score,mean_squared_error
#Load the data

data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
#Read the data 

data.head()
#Getting the data info

data.info()
#Analyse the missing value

missing_col = [col for col in data.columns if data[col].isnull().sum()>0]
missing_col
missing=data[missing_col].isnull().mean()
#Plot the missing data

plt.figure(figsize=(20,6))

sns.barplot(x = missing.index, y = missing.values)
#Analyse the missing values w.r.t target

def analayse_na(df,var):

    

    df = df.copy()

    df[var] = np.where(df[var].isnull(), 1, 0)

    df.groupby(var)['SalePrice'].median().plot.bar()

    

    plt.title(var)

    plt.show()
for var in missing_col:

    analayse_na(data,var)
#Numerical variable in dataset

num_vars = [col for col in data.columns if data[col].dtypes != 'O' and col not in 'Id']
num_vars
#Diagonastic plot to analyse the data

def analyse_dist(df, var):

    

    data[var].hist(bins=30)

    plt.ylabel('Number of houses')

    plt.xlabel(var)

    plt.title(var)

    plt.show()   
for var in num_vars:

    analyse_dist(data, var)
for var in num_vars:

    data.boxplot(column=var)

    plt.show()
#Categorical variable in dataset

cat_vars = [col for col in data.columns if data[col].dtypes == 'O']
cat_vars
#Analysing the cardinality of the categorical variable

def nunique_category(df, var):

    

    df[var].value_counts()

    df[var].value_counts().plot.bar()

    plt.xlabel(var)

    plt.ylabel('Number Counts')

    plt.show()
for cat in cat_vars:

    nunique_category(data, cat)
#Seggregating the data

X = data[[col for col in data.columns if col not in ['Id','SalePrice']]]

y = data['SalePrice']
#Split the data to training ad testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#Filling the missing numerical variable

num_missing_col = [col for col in missing_col if col in num_vars]
num_missing_col
data[num_missing_col].isnull().mean()*100
data[num_missing_col].describe()
#Impute the missing value for LotFrontage

lot_val=X_train['LotFrontage'].median()

X_train['LotFrontage'].fillna(value=lot_val, inplace=True)

X_test['LotFrontage'].fillna(value=lot_val, inplace=True)
#Impute the missing value for LotFrontage

garage_val = X_train['GarageYrBlt'].median()

X_train['GarageYrBlt'].fillna(value=garage_val, inplace=True)

X_test['GarageYrBlt'].fillna(value=garage_val, inplace=True)
#Impute Random Sample for MasVnrArea

impute_num = RandomSampleImputer(random_state=['MasVnrArea'], variables='MasVnrArea', seed='observation')

impute_num.fit(X_train)
X_train = impute_num.transform(X_train)

X_test = impute_num.transform(X_test)
#Filling the missing categorical columns

missing_cat = [col for col in missing_col if col in cat_vars]
data[missing_cat].isnull().mean()*100
#Drop the Alley, PoolQC, MiscFeature  column

X_train.drop(labels=['Alley', 'PoolQC', 'MiscFeature'], inplace=True, axis=1)

X_test.drop(labels=['Alley', 'PoolQC', 'MiscFeature'], inplace=True, axis=1)
data[missing_cat].describe()
variable_rsample = [col for col in missing_cat if data[col].isnull().mean()*100 < 3]
#Impute the Random Sample for 'MasVnrType','Electrical','BsmtQual','BsmtCond','BsmtFinType2',

#'BsmtExposure','BsmtFinType1'

impute_cat_r = RandomSampleImputer(variables=variable_rsample,

                                random_state=['MasVnrArea'],

                                seed='observation')
X_train = impute_cat_r.fit_transform(X_train)

X_test = impute_cat_r.transform(X_test)
variable_mcat = [col for col in missing_cat if data[col].isnull().mean()*100 > 3 if col not in 

                 ['Alley', 'PoolQC', 'MiscFeature']]
#Impute the 'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','Fence' missing variable by term 

#'Missing'



impute_cat_m = CategoricalVariableImputer(variables=variable_mcat)

impute_cat_m.fit(X_train)
X_train = impute_cat_m.transform(X_train)

X_test = impute_cat_m.transform(X_test)
num_vars.pop()
#Variable Transformation

vt  = PowerTransformer(variables=num_vars)

vt.fit(X_train,y_train)
#Transform the variable

X_train=vt.transform(X_train)

X_test=vt.transform(X_test)
#categorical variables after dropping some columns

cat_vars = [var for var in cat_vars if var not in ['Alley', 'PoolQC', 'MiscFeature']]
#Encode the Categorical variable as 'Rare' which occurs less than 1% of the observation

rare_encoder = RareLabelCategoricalEncoder(tol=0.01, variables=cat_vars)

rare_encoder.fit(X_train)
#transform the variable

X_train = rare_encoder.transform(X_train)

X_test = rare_encoder.transform(X_test)
#label encode the categorical variable

onehot_encoder = OneHotCategoricalEncoder(variables=cat_vars,drop_last=True)

onehot_encoder.fit(X_train,y_train)
#Transform the variable

X_train = onehot_encoder.transform(X_train)

X_test = onehot_encoder.transform(X_test)
#Fitting the DecisionTree model

for d in range(2,7):

    dtree = DecisionTreeRegressor(max_depth=d)

    dtree.fit(X_train,y_train)

    y_pred = dtree.predict(np.nan_to_num(X_test))

    print(f' For {d} depth  score for training  is {r2_score(y_train,dtree.predict(X_train))} and score for testing is {r2_score(y_test,y_pred)}')
#Fitting the RandomForest model

for n in [100,200,300,400,500,600,700]:

    rforest = RandomForestRegressor(n_estimators=n, max_depth=4)

    rforest.fit(X_train,y_train)

    y_pred = rforest.predict(np.nan_to_num(X_test))

    print(f' For {n} tree score for training  is {r2_score(y_train,rforest.predict(X_train))} and score for testing is {r2_score(y_test,y_pred)}')
#Fitting the Ridge model

for a in [0.1,1.0,2.0,3.0,4.0]:

    lasso=Lasso(alpha=a)

    lasso.fit(X_train,y_train)

    y_pred_l = lasso.predict(np.nan_to_num(X_test))

    print(f' For alpha {a}   score for training  is {r2_score(y_train,lasso.predict(X_train))} and score for testing is {r2_score(y_test,y_pred_l)}')
#Fitting the Ridge model

for a in [4.0,5.0,6.0,7.0,8.0]:

    ridge=Ridge(alpha=a)

    ridge.fit(X_train,y_train)

    y_pred_r = ridge.predict(np.nan_to_num(X_test))

    print(f' For alpha {a}   score for training  is {r2_score(y_train,ridge.predict(X_train))} and score for testing is {r2_score(y_test,y_pred_r)}')