import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model, metrics

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV



import os



# hide warnings

import warnings

warnings.filterwarnings('ignore')

#Reading dataset



df = train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.describe()
df.info()
# Checking null values in column % wise



round(100*(df.isnull().sum()/len(df.index)),2)[:60]
# Dropping Columns haviing null % >= 60%



df = df.drop(['Alley','PoolQC','Fence','MiscFeature'] , axis = 1)
#MSSubClass: Identifies the type of dwelling involved in the sale.



df.MSSubClass.value_counts()
# MSSubClass: Identifies the type of dwelling involved in the sale.



sns.distplot(df['MSSubClass'])

plt.show()
# LotArea: Lot size in square feet



sns.distplot(df['LotArea'])

plt.show()
#OverallQual: Rates the overall material and finish of the house



sns.distplot(df['OverallQual'])

plt.show()
#OverallCond : Rates the overall condition of the house



sns.distplot(df['OverallCond'])

plt.show()
# YearBuilt: Original construction date



sns.distplot(df['YearBuilt'])

plt.show()
# BsmtFinSF2: Type 2 finished square feet





sns.distplot(df['BsmtFinSF2'])

plt.show()
#1stFlrSF: First Floor square feet



sns.distplot(df['1stFlrSF'])

plt.show()
#2ndFlrSF: Second floor square feet



sns.distplot(df['2ndFlrSF'])

plt.show()
# LowQualFinSF: Low quality finished square feet (all floors)



sns.distplot(df['LowQualFinSF'])

plt.show()
sns.boxplot(data = df, x = 'MSZoning', y = 'SalePrice')
df['MSZoning'].hist()

plt.show()
df['BldgType'].hist()

plt.show()
#Few plots wrt target variable



fig = plt.figure(figsize = (15,10))



ax1 = fig.add_subplot(2,3,1)

sns.countplot(data = df, x = 'MSZoning', ax=ax1)



ax2 = fig.add_subplot(2,3,2)

sns.countplot(data = df, x = 'LotShape', ax=ax2)



ax3 = fig.add_subplot(2,3,3)

sns.countplot(data = df, x = 'LotConfig', ax=ax3)



ax4 = fig.add_subplot(2,3,4)

sns.boxplot(data = df, x = 'MSZoning', y = 'SalePrice' , ax=ax4)

#sns.violinplot(data = ds_cat, x = 'MSZoning', y = 'SalePrice' , ax=ax4)

#sns.swarmplot(data = ds_cat, x = 'MSZoning', y='SalePrice', color = 'k', alpha = 0.4, ax=ax4  )



ax5 = fig.add_subplot(2,3,5)

sns.boxplot(data = df, x = 'LotShape', y = 'SalePrice', ax=ax5)

#sns.violinplot(data = ds_cat, x = 'LotShape', y = 'SalePrice', ax=ax5)

#sns.swarmplot(data = ds_cat, x = 'LotShape', y='SalePrice', color = 'k', alpha = 0.4, ax=ax5  )



ax6 = fig.add_subplot(2,3,6)

sns.boxplot(data = df, x = 'LotConfig', y = 'SalePrice', ax=ax6)
# correlation matrix

cor = df.corr()

cor
# plotting correlations on a heatmap



# figure size

plt.figure(figsize=(40,20))



# heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()

100*(df.LotFrontage.value_counts(normalize = True)).head(60)
#Since, it is a continous variable, let's impute with mean / mode.



sns.boxplot(y = df['LotFrontage'])

plt.show()
#Outliers shows in the column , so perform metric analysis. Since, we are having outliers in this column so perfoming median



df.LotFrontage.median()

lotfrontage = df.LotFrontage.median()

df.LotFrontage.fillna(lotfrontage, inplace = True)
df.LotFrontage.isnull().sum()
100*(df.LotFrontage.value_counts(normalize = True))
# sns.boxplot(y=df['MasVnrType'])

# plt.show()



100*(df['MasVnrType'].value_counts(normalize = True))
df.MasVnrType.mode()
#Imputing with Mode



df['MasVnrType'].replace(np.nan, df['MasVnrType'].mode()[0], inplace=True)

100*(df['MasVnrType'].value_counts(normalize=True))
df.MasVnrType.isnull().sum()
df.MasVnrArea.value_counts().head(60)
df.MasVnrArea.isnull().sum()
# Imputing with mode



df['MasVnrArea'].replace(np.nan, df['MasVnrArea'].mode()[0], inplace=True)

100*(df['MasVnrArea'].value_counts(normalize=True))
df.MasVnrArea.isnull().sum()
#Binnig with other category



df['MasVnrArea'] = df['MasVnrArea'].where(df['MasVnrArea'].isin(df['MasVnrArea'].value_counts().index[:1]), 'Other')

100*(df['MasVnrArea'].value_counts(normalize=True))
df.BsmtQual.value_counts()
# replacing NA value with 'No basement'



df.BsmtQual.replace(np.nan, 'NoBasement', inplace=True)
#df.BsmtQual.value_counts()

100*(df['BsmtQual'].value_counts(normalize=True))
df.BsmtCond.value_counts()
# replacing NA value with 'No basement'



df.BsmtCond.replace(np.nan, 'NoBasement', inplace=True)
#df.BsmtCond.value_counts()

100*(df['BsmtQual'].value_counts(normalize=True))
df.BsmtExposure.value_counts()
# replacing NA value with 'No basement'



df.BsmtExposure.replace(np.nan, 'NoBasement', inplace=True)
df.BsmtExposure.value_counts()
df.BsmtFinType1.value_counts()
# replacing NA value with 'No basement'



df.BsmtFinType1.replace(np.nan, 'NoBasement', inplace=True)
#df.BsmtFinType1.value_counts()

100*(df['BsmtFinType1'].value_counts(normalize=True))
df.BsmtFinType2.value_counts()
# replacing NA value with 'No basement'



df.BsmtFinType2.replace(np.nan, 'NoBasement', inplace=True)
100*(df.BsmtFinType2.value_counts(normalize = True))

#Binning the variable with category 'Other'



df['BsmtFinType2'] = df['BsmtFinType2'].where(df['BsmtFinType2'].isin(df['BsmtFinType2'].value_counts().index[:1]), 'Other')

100*(df['BsmtFinType2'].value_counts(normalize=True))
df.GarageType.value_counts()
# replacing NA value with 'No basement'



df.GarageType.replace(np.nan, 'NoGarage', inplace=True)
100*(df.GarageType.value_counts(normalize = True))
# Binning the variable



df['GarageType'] = df['GarageType'].where(df['GarageType'].isin(df['GarageType'].value_counts().index[:2]), 'Other')

100*(df['GarageType'].value_counts(normalize=True))
df.GarageFinish.value_counts()
# replacing NA value with 'No basement'



df.GarageFinish.replace(np.nan, 'NoGarage', inplace=True)
df.GarageFinish.value_counts()
df.FireplaceQu.value_counts()
# replacing NA value with 'No basement'



df.FireplaceQu.replace(np.nan, 'NoFireplace', inplace=True)
df.FireplaceQu.value_counts()
# Binning the variable



df['FireplaceQu'] = df['FireplaceQu'].where(df['FireplaceQu'].isin(df['FireplaceQu'].value_counts().index[:3]), 'Other')

100*(df['FireplaceQu'].value_counts(normalize=True))
100*(df.Electrical.value_counts(normalize = True))
df.Electrical.isnull().sum()
# Imputing value with mode



df['Electrical'].replace(np.nan, df['Electrical'].mode()[0], inplace=True)

100*(df['Electrical'].value_counts(normalize=True))
#Since it is higly skewed , dropping the variable



df.drop(['Electrical'], axis = 1, inplace = True)
100*(df.GarageQual.value_counts(normalize = True))
#Since it is higly skewed , dropping variable



df.drop(['GarageQual'], axis = 1, inplace = True)
100*(df.GarageCond.value_counts(normalize = True))
#Since it is higly skewed , dropping it



df. drop(['GarageCond'], axis = 1, inplace = True)
100*(df.GarageYrBlt.value_counts(normalize = True))
df['GarageYrBlt'].hist()

plt.show()
df.GarageYrBlt.isnull().sum()
# Imputing value



df['GarageYrBlt'].replace(np.nan, df['GarageYrBlt'].mode()[0], inplace=True)

100*(df['GarageYrBlt'].value_counts(normalize=True))
#Histogram representation



df['GarageYrBlt'].hist()

plt.show()
#Checking distribution plot 



plt.figure(figsize=(10, 5))

sns.distplot(df['SalePrice'])

plt.show()
# Performing log transform to target variable



df['SalePrice_Log'] = np.log1p(df.SalePrice)

plt.figure(figsize=(10, 5))

sns.distplot(df['SalePrice_Log'])

plt.show()
#Dropping 'Saleprice' since it is of no use



df.drop(['SalePrice'], axis = 1, inplace = True)
df.head()
df.columns
# # Let's see the correlation matrix 



plt.figure(figsize = (40,20))        # Size of the figure

sns.heatmap(df.corr(),annot = True, cmap='Blues')

bottom,top = plt.ylim()

plt.ylim(bottom+0.5, top-0.5)

plt.show()
#dividing variable in X and y



X = df.loc[:,['MSSubClass','MSZoning','LotFrontage','LotArea','Street','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',

       'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',

       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',

       'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional',

       'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',

       'GarageFinish', 'GarageCars', 'GarageArea', 'PavedDrive', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']]



y = df['SalePrice_Log']
# Storing categorical variables in seprated dataframe



df_categorical = X.select_dtypes(include=['object'])

df_categorical.head()
df_dummies = pd.get_dummies(df_categorical, drop_first = True)

df_dummies.head()
# dropping unneccesary columns



X = X.drop(list(df_categorical.columns), axis = 1)
#Merging X and dummy variables



X = pd.concat([X, df_dummies], axis = 1)
X.shape
#variable YearBuilt_AGE



X['YearBuilt_AGE'] = X['YearBuilt'].apply(lambda x: 2020 - x)

X.YearBuilt_AGE.head()
#variable YearRemodAdd



X['YearRemodAdd_AGE'] = X['YearRemodAdd'].apply(lambda x: 2020 - x)

X.YearRemodAdd_AGE.head()
# Variable YrSold



X['YrSold_AGE'] = X['YrSold'].apply(lambda x: 2020 - x)

X.YrSold_AGE.head()
#Variable GarageYrBlt



X['GarageYrBlt_AGE'] = X['GarageYrBlt'].apply(lambda x: 2020 - x)

X.GarageYrBlt_AGE.head()
# Dropping all non-required valriables



X.drop(['YearBuilt','YearRemodAdd','YrSold','GarageYrBlt'], axis = 1, inplace = True)

X.head()
# # scaling the features

from sklearn.preprocessing import scale



# # # storing column names in cols

cols = X.columns

X = pd.DataFrame(scale(X))

X.columns = cols

X.columns
# # split into train and test



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,test_size = 0.3, random_state=100)
# from sklearn import preprocessing

# from sklearn import utils



# # lab_enc = preprocessing.LabelEncoder()

# # encoded = lab_enc.fit_transform(y_train)



# lab_enc = preprocessing.LabelEncoder()

# training_scores_encoded = lab_enc.fit_transform(y_train)

# print(training_scores_encoded)

# print(utils.multiclass.type_of_target(y_train))

# print(utils.multiclass.type_of_target(y_train.astype('int')))

# print(utils.multiclass.type_of_target(training_scores_encoded))
#Feature selection



from sklearn.linear_model import LinearRegression

logreg = LinearRegression()
from sklearn.feature_selection import RFE

rfe = RFE(logreg, 50)

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train_RFE = X_train[col]
# list of alphas to tune

params = {'alpha': [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]}





ridge = Ridge()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train_RFE, y_train)

#print(f"The best value of Alpha is: {model_cv.best_params_}")



print(f"The best value of Alpha is: {model_cv.best_params_}")
#Storing results in cv_results df



cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=200]

cv_results.head(60)
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
#Getting Final coefficients



alpha = 5.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train_RFE, y_train)

ridge.coef_
MergeDF_ridge = pd.DataFrame(ridge.coef_,index = col,columns=["Coeff"])

MergeDF_ridge.sort_values(by = ['Coeff'], ascending = False).head(10)
params = {'alpha' : [0.10, 0.2, 0.6, 2, 6, 10, 20]}

lasso = Lasso()



# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train_RFE, y_train)
print(f"The best value of Alpha is: {model_cv.best_params_}")
#Storing results



cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
#Getting final coefficient values



alpha = 0.1

lasso = Lasso(alpha=alpha)



lasso.fit(X_train_RFE, y_train)

lasso.coef_
MergeDF_Lasso = pd.DataFrame(lasso.coef_,index = col,columns=["Coeff"])

MergeDF_Lasso.sort_values(by=['Coeff'], ascending = False).head()
from sklearn.metrics import r2_score



r2_score(cv_results['mean_train_score'], cv_results['mean_test_score'])