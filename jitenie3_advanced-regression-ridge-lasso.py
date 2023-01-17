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
# Importing all required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import xticks

import seaborn as sns



%matplotlib inline
# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
# set the maximum display columns and rows

pd.set_option('display.max_columns', 111)

pd.set_option('display.max_rows', 50)
# Importing dataset

df = pd.read_csv("/kaggle/input/house-price-prediction/train.csv")

df.head()
df.tail()
df.info()
df.shape
# Column which contains null data

round(100*(df.isnull().sum()/len(df.index)), 2)[round(df.isnull().sum()/len(df.index), 2).values > 0.00].sort_values(ascending=False)
# Checking numeric column data

df.select_dtypes(include=['float64', 'int64']).describe()
# Convert year column to number or calculate the age for the column YearBuilt, YearRemodAdd, GarageYrBlt, YrSold

df['AgeYearBuilt'] = df.YearBuilt.max() - df['YearBuilt']

df['AgeYearRemodAdd'] = df.YearRemodAdd.max() - df['YearRemodAdd']

df['AgeGarageYrBlt'] = df.GarageYrBlt.max() - df['GarageYrBlt']

df['AgeYrSold'] = df.YrSold.max() - df['YrSold']



# drop the original column as we will use above created column

df.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'],axis=1,inplace=True)
df[['AgeYearBuilt', 'AgeYearRemodAdd', 'AgeGarageYrBlt', 'AgeYrSold']].head()
# Droping the column which have 

df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)
# No use of 'Id' column so droping it

df.drop(['Id'], axis=1, inplace=True)
# List of column still have empty data

round(100*(df.isnull().sum()/len(df.index)), 2)[round(df.isnull().sum()/len(df.index), 2).values > 0.00].sort_values(ascending=False)
# viewing data based on the interval percentage

df.describe(percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1])
# As per the above details few columns have standard value which we can use as categorical instead of numerical

df['MoSold'] = df['MoSold'].astype('object')

df['OverallQual'] = df['OverallQual'].astype('object')

df['OverallCond'] = df['OverallCond'].astype('object')

df['BsmtFullBath'] = df['BsmtFullBath'].astype('object')

df['BsmtHalfBath'] = df['BsmtHalfBath'].astype('object')

df['FullBath'] = df['FullBath'].astype('object')

df['HalfBath'] = df['HalfBath'].astype('object')

df['BedroomAbvGr'] = df['BedroomAbvGr'].astype('object')

df['KitchenAbvGr'] = df['KitchenAbvGr'].astype('object')

df['TotRmsAbvGrd'] = df['TotRmsAbvGrd'].astype('object')

df['Fireplaces'] = df['Fireplaces'].astype('object')

df['GarageCars'] = df['GarageCars'].astype('object')
# Column which contains outliers 

out_col = [

    'LotArea',

    'TotalBsmtSF',

    'PoolArea',

    'MiscVal']
# Boxplot method to generate the graph to Check the outliers 



def draw_boxplot(cols):

    int_range = range(len(cols))[::3]

    col_length = len(cols)

    for col in int_range:

        print('----------------',cols[col:col+3],' ----------------')

        plt.figure(figsize=(17, 5))

        if col < col_length:  

            plt.subplot(1,3,1)

            sns.boxplot(x=cols[col], orient='v', data=df)

        if col+1 < col_length:                    

            plt.subplot(1,3,2)

            sns.boxplot(x=cols[col+1], orient='v', data=df)

        if col+2 < col_length:                

            plt.subplot(1,3,3)

            sns.boxplot(x=cols[col+2], orient='v', data=df)

                        

        plt.show()
# Method call to draw boxplot for the outliers

draw_boxplot(out_col)
# Size before removing the outliers

df.shape
# method to remove outliers

def remove_outliers(x, num_cols, s_quntl=0.05, e_quntl=0.95):

    for col in num_cols:

        Q1 = x[col].quantile(s_quntl)

        Q3 = x[col].quantile(e_quntl)

        IQR = Q3-Q1

        x =  x[(x[col] >= (Q1-(1.5*IQR))) & (x[col] <= (Q3+(1.5*IQR)))] 

    return x   
# call remove outliers method for the selected columns

df=remove_outliers(df, out_col)
# dataframe size after removing the outliers

df.shape
df.describe(percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
# Method to replace the null value with the selected values

def filling_missing_values(col, replace_type:str, other_value=None):

    if replace_type == 'mean':

        df[col].fillna(df[col].mean(), inplace=True)  

    if replace_type == 'mode':

        df[col].fillna(df[col].mode()[0], inplace=True)

    if replace_type == 'median':

        df[col].fillna(df[col].median(), inplace=True)

    if replace_type == 'other':

        df[col].fillna(other_value, inplace=True)

# list of columns which contains null value

null_cols = df.columns[round(df.isnull().sum()/len(df.index), 2).values > 0.00]

null_cols
# column list which has null value

df[null_cols].describe(percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
# Categorical value updating using mode values

for col in null_cols:

    if col not in ['LotFrontage', 'AgeGarageYrBlt', 'MasVnrArea']:

        filling_missing_values(col, 'mode')



# updating with mean value for the variable MasVnrArea   

filling_missing_values('LotFrontage', 'mean') 

filling_missing_values('MasVnrArea', 'mean') 

filling_missing_values('AgeGarageYrBlt', 'other', other_value=0)

# Check the column which still have null values

round(100*(df.isnull().sum()/len(df.index)), 2)[round(df.isnull().sum()/len(df.index), 2).values > 0.00].sort_values(ascending=False)
num_cols=df.select_dtypes(include=['int64', 'float']).columns

num_cols
# Method to fetch column list which contains more than 90% duplicate value

def percentage_of_duplicate(num_cols):

    x=list()

    for col in (num_cols):

        if(df[col].value_counts().max()/df.shape[0] >= 0.90):

            x.append(col)

    return x
# drop filtered column

filter_cols=percentage_of_duplicate(num_cols)

print(filter_cols)

df.drop(filter_cols, axis = 1, inplace = True)
# target variable SalePrice

plt.figure(figsize=(15,5))

plt.title('SalePrice')

sns.distplot(df.SalePrice)

plt.show()
plt.title('SalePrice')

sns.distplot(np.log1p(df['SalePrice']), bins=10)

plt.show()
from scipy import stats

stats.probplot(df['SalePrice'], plot=plt)

plt.show()
# Method to viewing all the categorical variable

def categorical_data(cols):

    for col in cols:

        print('\n')

        print('---------------------------------------------- ',col,' -----------------------------------------------')

        print(df[col].astype('category').value_counts())

        f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(12,3), dpi=90) 

        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax1) 

        ax1.set_ylabel('Count') 

        ax1.set_title(f'{col}', weight="bold") 

        ax1.set_xlabel(col) 

        if col == 'Neighborhood':

            xticks(rotation = 90)

        plt.show()

# list of categorical columns

categ_var = df.select_dtypes(include=['object']).columns

# Visualise the data

categorical_data(categ_var)
# As per the above graph Street and utilities has lower variance so dorpping it

df.drop(['Street','Utilities'],axis=1,inplace=True)
# Numerical variable analysis using pairplots

def numerical_data(cols):

    int_range = range(len(cols))[::3]

    col_length = len(cols)

    for col in int_range:

        print('------------------ ',cols[col:col+3],' ---------------------')

        sns.pairplot(df, x_vars=cols[col:col+3], y_vars='SalePrice',height=3, aspect=1,kind='scatter')            

        plt.show()

# List of numeric columns 

num_cols=df.select_dtypes(include=['int64', 'float']).columns

num_cols
num_cols = num_cols.drop(labels='SalePrice')

num_cols
# Check the numerical values using pairplots



# Target variable SalePrice and other variables

numerical_data(num_cols)
# correlation table to check the correlation for the variable with others

df.corr()
# Heatmap to check correlatoin between the variables 

fig, ax = plt.subplots() 

fig.set_size_inches(35, 30) 

sns.heatmap(df.corr(),cmap ="YlGnBu",linewidths = 0.1, annot = True)

top, bottom = ax.get_ylim()

ax.set_ylim(top+0.5, bottom-0.5)

plt.show()
# positive correlation with SalePrice greater than 50%

corr = df.corr()

top_feature = corr.index[abs(corr['SalePrice']>0.5)]

fig, ax = plt.subplots() 

fig.set_size_inches(15, 10) 

top_corr = df[top_feature].corr()

sns.heatmap(top_corr,cmap ="YlGnBu",linewidths = 0.1, annot = True)

top, bottom = ax.get_ylim()

ax.set_ylim(top+0.5, bottom-0.5)

plt.show()
# negative correlation with SalePrice less then -0.5

corr = df.corr()

top_feature = corr.index[abs(corr['SalePrice']<-0.5)]

fig, ax = plt.subplots() 

fig.set_size_inches(10, 5) 

top_corr = df[top_feature].corr()

sns.heatmap(top_corr,cmap ="YlGnBu",linewidths = 0.1, annot = True)

top, bottom = ax.get_ylim()

ax.set_ylim(top+0.5, bottom-0.5)

plt.show()
# Dividing dataframe into X and Y sets for the model building

X=df.drop(columns=['SalePrice'])

y=np.log(df['SalePrice'])
# Viewing categorical data

categorical_data = X.select_dtypes(include=['object'])

categorical_data.head(3)
# Use pandas library to create the dummy variables

dummies = pd.get_dummies(categorical_data, drop_first=True)

dummies.head(15)
# drop categorical data for that dummy variable has created

X=X.drop(columns=categorical_data)

X.head(3)
# concat dummies with the X numerical variable

X=pd.concat([X,dummies],axis=1)

X.head(3)
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
# Train and test data shape

print(X_train.shape)

print(X_test.shape)
from sklearn.preprocessing import StandardScaler
num_col=X_train.select_dtypes(include=['int64','float64']).columns

num_col
# Apply scaler() to all the columns except the dummy variables which we creaeted before

scaler = StandardScaler()

X_train[num_col] = scaler.fit_transform(X_train[num_col])

X_test[num_col] = scaler.transform(X_test[num_col])
X_train.head(3)
# Importing RFE and LinearRegression from the sklearn

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable 

lm = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm, 25)

rfe = rfe.fit(X_train, y_train)
rfe_df = pd.DataFrame(list(zip(X_train.columns,rfe.support_,rfe.ranking_)), columns=['Variable', 'rfe_support', 'rfe_ranking'])

rfe_df = rfe_df.loc[rfe_df['rfe_support'] == True]

rfe_df.reset_index(drop=True, inplace=True)

rfe_df
# Selected column list 

col = X_train.columns[rfe.support_]

col
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_rfe.head()
# import Ridge and Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

# from sklearn.metrics import mean_squared_error
# list of alphas to tune our model

params = {'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
# Ridge 

ridge = Ridge()
# Cross Validation 

folds = 5

RidgeModelCV = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)           

RidgeModelCV.fit(X_train, y_train) 
#checking the value of optimum number of parameters

print(RidgeModelCV.best_params_)

print(RidgeModelCV.best_score_)
# Result based on the mean score



RidgeModelCVResults = pd.DataFrame(RidgeModelCV.cv_results_)

RidgeCVResults = RidgeModelCVResults[RidgeModelCVResults['param_alpha']<=200]

RidgeCVResults[['param_alpha', 'mean_train_score', 'mean_test_score', 'rank_test_score']].sort_values(by = ['rank_test_score'])


RidgeCVResults['param_alpha'] = RidgeCVResults['param_alpha'].astype('int32')

# plotting mean for train and test score with alpha 

RidgeCVResults['param_alpha'] = RidgeCVResults['param_alpha']

plt.figure(figsize=(16,5))

plt.plot(RidgeCVResults['param_alpha'], RidgeCVResults['mean_train_score'])

plt.plot(RidgeCVResults['param_alpha'], RidgeCVResults['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
RidgeModelCV.best_estimator_
print(int(RidgeModelCV.best_params_.get('alpha')))
alpha = int(RidgeModelCV.best_params_.get('alpha'))

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
#lets predict the R-squared value of test and train data

y_train_pred = ridge.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = ridge.predict(X_test)

print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
# RMSE

metrics.mean_squared_error(y_test, ridge.predict(X_test))
alpha = int(RidgeModelCV.best_params_.get('alpha'))*2

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_

y_train_pred = ridge.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = ridge.predict(X_test)

print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
# RMSE

metrics.mean_squared_error(y_test, ridge.predict(X_test))
ridge_df = pd.DataFrame({'Features':X_train.columns, 'Coefficient':ridge.coef_.round(4)})

ridge_df.reset_index(drop=True, inplace=True)

ridge_df
# convert in dict for other usages

ridge_coeff = dict(pd.Series(ridge.coef_.round(4), index = X_train.columns))
# minimise the feature using RFE

X_train_ridge = X_train[ridge_df.Features]



lm = LinearRegression()

lm.fit(X_train_ridge, y_train)



rfe = RFE(lm, 15)            

rfe = rfe.fit(X_train_ridge, y_train)
ridge_df1 = pd.DataFrame(list(zip( X_train_ridge.columns, rfe.support_, rfe.ranking_)), columns=['Features', 'rfe_support', 'rfe_ranking'])

ridge_df1 = ridge_df1.loc[ridge_df1['rfe_support'] == True]

ridge_df1.reset_index(drop=True, inplace=True)



ridge_df1['Coefficient'] = ridge_df1['Features'].apply(lambda x: ridge_coeff[x])

ridge_df1 = ridge_df1.sort_values(by=['Coefficient'], ascending=False)

ridge_df1 = ridge_df1.head(10)

ridge_df1
plt.figure(figsize=(15,5))

sns.barplot(y = 'Features', x='Coefficient', data = ridge_df1)

plt.show()
lasso = Lasso()



# list of alphas

params = {'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01]}



# cross validation

folds = 5

LassoModelCV = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)             



LassoModelCV.fit(X_train, y_train) 
#checking the value of optimum number of parameters

print(LassoModelCV.best_params_)

print(LassoModelCV.best_score_)
# display the mean scores



LassoModelCVResults = pd.DataFrame(LassoModelCV.cv_results_)

LassoModelCVResults[['param_alpha', 'mean_train_score', 'mean_test_score', 'rank_test_score']].sort_values(by = ['rank_test_score'])
LassoModelCVResults['param_alpha'] = LassoModelCVResults['param_alpha'].astype('float32')



# plotting mean for train and test score with alpha 



plt.figure(figsize=(16,5))

plt.plot(LassoModelCVResults['param_alpha'], LassoModelCVResults['mean_train_score'])

plt.plot(LassoModelCVResults['param_alpha'], LassoModelCVResults['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
LassoModelCV.best_params_.get('alpha')
alpha = LassoModelCV.best_params_.get('alpha')

lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train) 

lasso.coef_
#lets predict the R-squared value of test and train data

y_train_pred = lasso.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lasso.predict(X_test)

print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
# RMSE

metrics.mean_squared_error(y_test, lasso.predict(X_test))
alpha = LassoModelCV.best_params_.get('alpha')*2

lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train) 

lasso.coef_
y_train_pred = lasso.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lasso.predict(X_test)

print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
metrics.mean_squared_error(y_test, lasso.predict(X_test))
lasso_df = pd.DataFrame({'Features':X_train.columns, 'Coefficient':lasso.coef_.round(4)})

lasso_df.reset_index(drop=True, inplace=True)

lasso_df
lasso_coeff = dict(pd.Series(lasso.coef_.round(4), index = X_train.columns))

# minimise the feature using RFE

X_train_lasso = X_train[lasso_df.Features]



lm = LinearRegression()

lm.fit(X_train_lasso, y_train)



rfe = RFE(lm, 15)            

rfe = rfe.fit(X_train_lasso, y_train)
lasso_df = pd.DataFrame(list(zip( X_train_lasso.columns, rfe.support_, rfe.ranking_)), columns=['Features', 'rfe_support', 'rfe_ranking'])

lasso_df = lasso_df.loc[lasso_df['rfe_support'] == True]

lasso_df.reset_index(drop=True, inplace=True)



lasso_df['Coefficient'] = lasso_df['Features'].apply(lambda x: lasso_coeff[x])

lasso_df = lasso_df.sort_values(by=['Coefficient'], ascending=False)

lasso_df = lasso_df.head(10)

lasso_df
plt.figure(figsize=(15,5))

sns.barplot(y = 'Features', x='Coefficient', data = lasso_df)

plt.show()