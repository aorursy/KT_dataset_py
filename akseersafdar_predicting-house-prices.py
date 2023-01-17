#making the imports



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import Image

import seaborn as sns

sns.set_style('dark')

import os

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#print the directory items

print(os.listdir('../input/house-prices-advanced-regression-techniques'))
#read the test and train files

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(df_train.head())

print('***************************************************\n')

print(df_test.head())
#checking the shape of train and test set



print('The shape of training set is: {}'.format(df_train.shape))

print('\n')

print('The shape of test set is: {}'.format(df_test.shape))
#assigning id column to this variable. we will use it later when writing the submission file. 

test_ID = df_test['Id']



#delete the id column from datasets

del df_train['Id']

del df_test['Id']
#making the scatter plot

def scater_plot(x,y,x2,y2):

    plt.figure(figsize = (10,8))

    sns.scatterplot(x,y)

    plt.xlabel(x2,fontsize = 15)

    plt.ylabel(y2,fontsize = 15)

    plt.show()
#scatter plot for living area vs sale price



scater_plot(df_train['GrLivArea'], df_train['SalePrice'], 'GrLivArea', 'SalePrice')
#scatter plot of Total basement vs sale price



scater_plot(df_train['TotalBsmtSF'], df_train['SalePrice'], 'TotalBsmtSF', 'SalePrice')
#scatter plot of Garage Area vs sale price



scater_plot(df_train['GarageArea'], df_train['SalePrice'], 'GarageArea', 'SalePrice')
#scatter plot of Lot Area vs sale price



scater_plot(df_train['LotArea'], df_train['SalePrice'], 'LotArea', 'SalePrice')
#remove the outliers and store data in a temp df to re-visualize the relation between

#lot area and sales price. Now we see bit of a linear relationship



temp_df = df_train[df_train['LotArea'] < 40000]

scater_plot(temp_df['LotArea'], temp_df['SalePrice'], 'LotArea', 'SalePrice')
#making the box plot for over all quality vs the sale price



plt.figure(figsize = (14,8))

sns.boxplot(df_train['OverallQual'], df_train['SalePrice'], palette = 'pastel')

plt.xlabel('OverallQual', fontsize = 15)

plt.ylabel('SalePrice', fontsize = 15)

plt.title('Over all Quality vs Sale Price', fontsize = 20)

plt.show()
#making the box plot for year built vs the sale price



plt.figure(figsize = (22,8))

sns.boxplot(df_train['YearBuilt'], df_train['SalePrice'], palette = 'pastel')

plt.xlabel('YearBuilt', fontsize = 15)

plt.ylabel('SalePrice', fontsize = 15)

plt.title('Year Built vs Sale Price', fontsize = 20)

plt.axis(ymin=0, ymax=600000);

plt.xticks(rotation=90)

plt.show()
#correlation matrix for all features in training set



plt.figure(figsize = (22,18))

corrmat = df_train.corr()

sns.heatmap(corrmat, vmax=.8, square=True, cmap= 'viridis')

plt.show()
#lets visualize the correlation between less variable which are more correlated



plt.figure(figsize = (18,14))

corr_matrix = df_train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageYrBlt', 'GarageArea', 'TotalBsmtSF',

                       '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']].corr()



sns.heatmap(corr_matrix, vmax = 0.8, linewidths= 0.01, square= True, 

           annot= True, cmap= 'viridis', linecolor= 'white')



plt.title('Correlation Matrix')

plt.show()
#check the distribution of target variable 



print ("Skew is:", df_train.SalePrice.skew())

print('\n')

plt.figure(figsize = (8,6))

plt.hist(df_train.SalePrice, color='blue')

plt.show()
#making the distribution plot for sale price



plt.figure(figsize = (8,6))

sns.distplot(df_train['SalePrice'])

plt.xlabel('Sale Price', fontsize = 15)

plt.show()
#checking for numerical and categorical features



numerical_feats = df_train.dtypes[df_train.dtypes != 'object'].index

print("Number of Numerical features: ", len(numerical_feats))



categorical_feats = df_train.dtypes[df_train.dtypes == 'object'].index

print("Number of Categorical features: ", len(categorical_feats))
#printing the columns



print(df_train[numerical_feats].columns)

print("*"*80)

print("*"*80)

print(df_train[categorical_feats].columns)
#count and percent of missing values



total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Percent']>0]
# Function for value counts in each columns



def cat_exploration(column):

    return df_train[column].value_counts()



# Function for Imputing the missing values



def cat_imputation(column, value):

    df_train.loc[df_train[column].isnull(),column] = value
#A number of values are missing and one possibility would be to just impute the mean. 

#However, there should actually be a correlation with LotArea, which has no missing values.

#check correlation of LotFrontage with LotArea



df_train['LotFrontage'].corr(df_train['LotArea'])
# we assume that most lots are rectangular, using the square root might be an improvement.



df_train['SqrtLotArea']=np.sqrt(df_train['LotArea'])

df_train['LotFrontage'].corr(df_train['SqrtLotArea'])
#see the pair plot for LotFrontage and SqrtLotArea



sns.pairplot(df_train[['LotFrontage','SqrtLotArea']].dropna())

plt.show()
#imputing the missing values in LotFrontage column



cond = df_train['LotFrontage'].isnull()

df_train.LotFrontage[cond]=df_train.SqrtLotArea[cond]
# This column is not needed now. lets delete it. 



del df_train['SqrtLotArea']
#value counts for  Alley column



cat_exploration('Alley')
# empty fields just means that there is no alley access so replace it with none



cat_imputation('Alley','None')
#value counts for MasVnrType



cat_exploration('MasVnrType')
#fill the missing with None



cat_imputation('MasVnrType', 'None')
#value counts MasVnrArea

cat_exploration('MasVnrArea')
#0.0 is the most frequent value

cat_imputation('MasVnrArea', 0.0)
#doing the imputation for basement related columns



basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

df_train[basement_cols][df_train['BsmtQual'].isnull()==True]
#replace Nan with None



for cols in basement_cols:

    if 'FinSF'not in cols:

        cat_imputation(cols,'None')
#value counts for  Electrical column

cat_exploration('Electrical')
#impute with the most occuring value



cat_imputation('Electrical', 'SBrkr')
#value counts for  fireplaceQu

cat_exploration('FireplaceQu')
#lets check the count of missing values



df_train['FireplaceQu'].isnull().sum()
#impute with none as these don't have a fireplace



cat_imputation('FireplaceQu','None')
#making corsstab with Fireplaces column



pd.crosstab(df_train.Fireplaces, df_train.FireplaceQu)
#now lets do it for Garages columns



garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

df_train[garage_cols][df_train['GarageType'].isnull()==True]
#Garage Imputation - zero for numerical and none for categorical columns



for cols in garage_cols:

    if df_train[cols].dtype==np.object:

        cat_imputation(cols,'None')

    else:

        cat_imputation(cols, 0)
#value counts for PoolQC

cat_exploration('PoolQC')
#count of missing values

df_train['PoolQC'].isnull().sum()
#seems like the missing are the ones where there is no pool in the house. lets put none there



cat_imputation('PoolQC', 'None')
#value counts for Fence column

cat_exploration('Fence')
#seems like missing ones are the ones with no Fence

cat_imputation('Fence', 'None')
#value counts for MiscFeatures

cat_exploration('MiscFeature')
#the missing ones are the ones where we don't have these features

cat_imputation('MiscFeature', 'None')
#Let's check if we still have missing values in training set



total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head()
#we are done imputing the missing values :-)



df_train.isnull().any()
#count and percent of missing values



total = df_test.isnull().sum().sort_values(ascending=False)

percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Percent']>0]
# We will use the same two functions for value counts of a column and also for imputation. 

def cat_exploration(column):

    return df_test[column].value_counts()



# Function for Imputing the missing values



def cat_imputation(column, value):

    df_test.loc[df_test[column].isnull(),column] = value
# as we have already imputed missing values in training set

# lets do it the same way for test set for some columns which are matching 



cat_imputation('Alley','None')

cat_imputation('MasVnrType', 'None')

cat_imputation('MasVnrArea', 0.0)

cat_imputation('FireplaceQu','None')

cat_imputation('PoolQC', 'None')

cat_imputation('Fence', 'None')

cat_imputation('MiscFeature', 'None')
#imputation for basement columns



basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']



#replace Nan with None

for cols in basement_cols:

    if 'FinSF'not in cols:

        cat_imputation(cols,'None')
#imputation for garage columns



garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']



#Garage Imputation - zero for numerical and none for categorical columns

for cols in garage_cols:

    if df_test[cols].dtype==np.object:

        cat_imputation(cols,'None')

    else:

        cat_imputation(cols, 0)
#count and percent of missing values



total = df_test.isnull().sum().sort_values(ascending=False)

percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Percent']>0]
#A number of values are missing and one possibility would be to just impute the mean. 

#However, there should actually be a correlation with LotArea, which has no missing values.

# check correlation of LotFrontage with LotArea



df_test['LotFrontage'].corr(df_test['LotArea'])
# we assume that most lots are rectangular, using the square root might be an improvement.



df_test['SqrtLotArea']=np.sqrt(df_test['LotArea'])

df_test['LotFrontage'].corr(df_test['SqrtLotArea'])

#see the pair plot for LotFrontage and SqrtLotArea



sns.pairplot(df_test[['LotFrontage','SqrtLotArea']].dropna())

plt.show()

#imputing the missing values in LotFrontage column



cond = df_test['LotFrontage'].isnull()

df_test.LotFrontage[cond]=df_test.SqrtLotArea[cond]

# This column is not needed now



del df_test['SqrtLotArea']
#count and percent of missing values



total = df_test.isnull().sum().sort_values(ascending=False)

percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Percent']>0]
#value counts for total basement SF column



cat_exploration('TotalBsmtSF')
#imputation for numerical columns



num_cols = ['BsmtHalfBath','BsmtFullBath', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFinSF1', 'TotalBsmtSF']



#replace with most frequent value which is 0.0



for col in num_cols:

    cat_imputation(col,'0.0')

    

#count and percent of missing values



total = df_test.isnull().sum().sort_values(ascending=False)

percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Percent']>0]
#imputation for catrgory columns



cat_cols =['MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'SaleType', 'Exterior2nd', 'KitchenQual']



#replace with none

for col in cat_cols:

    cat_imputation(col,'None')
#count and percent of missing values



total = df_test.isnull().sum().sort_values(ascending=False)

percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Percent']>0]
#we are done imputing the missing values in test set.



df_test.isnull().any()
#lets put all features other than sale price in a list



features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',

       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',

       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',

       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',

       'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
#create X and y to be used for training the model



X = df_train[features]

y = df_train.SalePrice
#divide the training set into train/validation set with 20% set aside for validation. 



from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=101)
#as we know that we can give categorical features to catboost to make best use of its performance. 

#all categorical features will be where data type is not float



categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
#importing catboost regressor and define key parameters

#we can play with some parameters like learning rate etc



from catboost import CatBoostRegressor

model=CatBoostRegressor(iterations=1000, 

                        depth=5, 

                        learning_rate=0.1,

                        loss_function='RMSE',

                        random_seed=1,

                        bagging_temperature=22,

                        od_type='Iter',

                        metric_period=100,

                        od_wait=100)
#train the model (1000 iterations)



model.fit(X_train, y_train,cat_features=categorical_features_indices,

          eval_set=(X_validation, y_validation),plot=True)
#as we can see that the model starts to overfit after 100 iterations

#so lets train a new model for 100 iterations only



model_2=CatBoostRegressor(iterations=100, 

                        depth=5, 

                        learning_rate=0.1,

                        loss_function='RMSE',

                        random_seed=1,

                        bagging_temperature=22,

                        od_type='Iter',

                        metric_period=100,

                        od_wait=100)



#train the model (100 iterations)



model_2.fit(X_train, y_train,cat_features=categorical_features_indices,

          eval_set=(X_validation, y_validation),plot=True)
#make the submission file



submission_02 = pd.DataFrame()

submission_02['Id'] = test_ID

submission_02['SalePrice'] = model.predict(df_test)

submission_02.to_csv("Submission_02.csv", index=False)