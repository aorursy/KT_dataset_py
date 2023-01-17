# Importing libraries



# Overall libraries

import pandas as pd

import numpy as np

import statsmodels.api as sm



# Plotting libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# scikit learnit

import sklearn

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
# Setting our environment, we want to be able to see all the rows and all the columns, specially for the correlation matrix we will build



pd.set_option("max_columns", None)

pd.set_option("max_rows", None)
# Importing DataSet



df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_train.head()
# Quick view of the DataFrame's description, specially total rows, the type of each variable and the number of non-null values.



df_train.info()
# Creating a null rank

null_count = df_train.isnull().sum().sort_values(ascending=False)

null_percentage = null_count / len(df_train)

null_rank = pd.DataFrame(data=[null_count, null_percentage],index=['null_count', 'null_ratio']).T

null_rank
df_train.PoolArea.value_counts()
# dropping columns with over 70% missing values and PoolArea, which doesn't have actual values, most of it is zero.



df_drop = df_train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'PoolArea'], axis = 1)

df_drop.head()
# Let's check the numeric attributes of this dataframe



df_drop.describe().round(2)
# Let's check how the attributes distance themselves on the x axis

df_drop.hist(bins = 50, figsize = (20,15))

plt.show()
# Creating a copy of the original dataframe



housing = df_drop.copy()
# Creating a correlation matrix



corr_matrix = housing.corr()
# Checking the correlation to our target attribute, sale price



corr_matrix['SalePrice'].sort_values(ascending = False)
# Let's remove the variables with less than 10% correlation



housing = housing.drop(['MoSold', '3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'Id', 'LowQualFinSF', 'YrSold',

                        'OverallCond', 'MSSubClass'], axis = 1)
# Creating a new correlation matrix, since we dropped a few columns



corr_matrix = housing.corr()
# Checking the correlation to our target, the Sale Price



corr_matrix['SalePrice'].sort_values(ascending = False)
# Creating a list of numeric columns and categorical columns



num_list = list(housing.select_dtypes(include = [np.int, np.float]).columns)

cat_list = list(housing.select_dtypes(include = [object]).columns)
# Creating a numeric DataFrame



df_num = housing.drop(cat_list, axis = 1)

df_num.shape
# Defining the imputer



imputer = SimpleImputer(strategy='mean')
# Fitting and transforming the imputer



housing_num = pd.DataFrame(imputer.fit_transform(df_num), columns = num_list)
# Checking if there were any missing values behind



housing_num.isnull().any().sum()
# describing number variables



housing_num.describe().round(2)
# Creating a categorical DataFrame



df_cat = housing.drop(num_list, axis = 1)
# Creating dummies for all the categorical variables.



cat_dummies = pd.get_dummies(df_cat)

cat_dummies.shape
# Creating the X variable



X = pd.concat([housing_num, cat_dummies], axis = 1, join = 'outer')

X.isnull().any().sum()
# Removing SalePrice and creating our target variable



y = X['SalePrice']

X = X.drop(['SalePrice'], axis = 1)
# Creating the scaler



scaler = StandardScaler()
# Let's scale X



X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

X_scaled.describe().round(2)
# Let's log y the variables



y = np.log(y)
# Let's see some statistic magic



X1 = sm.add_constant(X_scaled)

est = sm.OLS(y, X1)

est1 = est.fit()

print(est1.summary())
# Let's define an y_pred to calculate what a probable mse and r² would be



y_pred = est1.predict(X1)

# Let's calculate the MSE



mse = sklearn.metrics.mean_squared_error(y, y_pred)

mse
# Let's calculate the RMSE



rmse = mse**(1/2)

rmse
# Let's calculate the r²



sklearn.metrics.r2_score(y, y_pred)
# Creating the train  test set



X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size = 0.3, random_state = 42)
# Creating the Linear Regression and fitting it



lr = LinearRegression(fit_intercept = False)

lr.fit(X_train, y_train)
# Creating the prediction of y



y_pred = lr.predict(X_test)
# Checking the adherence of the model so far



_ = plt.figure(figsize=(8,8))

_ = sns.regplot(x = y_test, y = y_pred)
# Let's calculate the MSE



mse = sklearn.metrics.mean_squared_error(y_test, y_pred)

mse
# Let's calculate the RMSE



rmse = mse**(1/2)

rmse
# Let's calculate the r²



sklearn.metrics.r2_score(y_test, y_pred)
# Creating an unified base



unified = pd.concat([X, y], axis = 1)
# Creating a correlation matrix



corr_matrix = unified.corr()
# Checking the correlation to SalePrice



corr_matrix['SalePrice'].sort_values(ascending = False)
list_to_drop = ['RoofStyle_Hip', 'Neighborhood_StoneBr', 'Neighborhood_Somerst', 'LotConfig_CulDSac', 'BsmtExposure_Av', 'Neighborhood_Timber',

                'BsmtCond_TA', 'Heating_GasA', 'Functional_Typ', 'BldgType_1Fam', 'ExterCond_TA', 'LotShape_IR2', 'BsmtFinType2_Unf', 

                'ScreenPorch', 'MSZoning_FV', 'RoofMatl_WdShngl', 'Condition1_Norm', 'Neighborhood_CollgCr', 'LandContour_HLS', 'BsmtCond_Gd',

                'Exterior1st_CemntBd', 'Exterior2nd_CmentBd', 'Neighborhood_Crawfor', 'Neighborhood_Gilbert', 'Neighborhood_ClearCr', 

                'Neighborhood_Veenker', 'Condition1_PosN', 'Neighborhood_NWAmes', 'Street_Pave', 'RoofMatl_WdShake', 'Condition1_PosA',

                'BsmtExposure_Mn', 'GarageQual_Gd', 'Condition2_Norm', 'Exterior2nd_ImStucc', 'Condition2_PosA', 'Condition2_PosN', 

                'SaleType_Con', 'Exterior2nd_Other', 'BsmtFinType2_ALQ', 'Exterior1st_Stone', 'Neighborhood_Blmngtn', 'LandContour_Low',

                'LotShape_IR3', 'Neighborhood_SawyerW', 'HouseStyle_2.5Fin', 'Exterior1st_BrkFace', 'Exterior1st_ImStucc', 'LandSlope_Mod',

                'RoofStyle_Shed', 'BldgType_TwnhsE', 'LandSlope_Sev', 'RoofMatl_Membran', 'RoofStyle_Flat', 'SaleType_CWD', 'Condition1_RRNn',

                'LotConfig_FR3', 'GarageQual_Ex', 'Condition1_RRAn', 'Exterior2nd_BrkFace', 'Utilities_AllPub', 'Condition1_RRNe',

                'Exterior1st_Plywood', 'ExterCond_Ex', 'RoofMatl_Tar&Grv', 'Foundation_Wood', 'Condition2_RRAe', 'RoofStyle_Mansard', 

                'GarageCond_Gd', 'RoofMatl_Metal', 'LotConfig_FR2', 'LotConfig_Corner', 'SaleType_ConLI', 'BsmtFinType2_GLQ', 

                'RoofMatl_ClyTile', 'FireplaceQu_Fa', 'LandContour_Lvl','HouseStyle_SLvl', 'Utilities_NoSeWa', 'RoofMatl_Roll',

                'Condition2_RRAn', 'Foundation_Stone', 'Functional_Sev', 'Neighborhood_Blueste', 'Exterior2nd_Stone', 'GarageType_2Types', 'BsmtFinType2_LwQ',

                'Exterior2nd_Plywood', 'Exterior2nd_AsphShn', 'SaleCondition_Alloca', 'HouseStyle_2.5Unf', 'Heating_GasW', 'Heating_OthW', 'GarageCond_Ex',

                'BsmtFinType2_Rec', 'Exterior2nd_CBlock', 'Exterior1st_CBlock', 'GarageType_Basment', 'Neighborhood_NPkVill', 'Exterior1st_AsphShn', 'LandSlope_Gtl',

                'SaleType_ConLw', 'SaleType_Oth', 'Functional_Maj1', 'Neighborhood_Mitchel', 'Condition2_Artery', 'Functional_Mod', 'HeatingQC_Po', 'MasVnrType_BrkCmn',

                'Exterior1st_Stucco', 'Condition1_RRAe', 'SaleCondition_Family', 'ExterCond_Gd', 'RoofStyle_Gambrel', 'SaleType_ConLD', 'Exterior2nd_HdBoard',

                'ExterCond_Po', 'BsmtFinType2_BLQ', 'Heating_Floor', 'Condition2_RRNn', 'Condition2_Feedr', 'Street_Grvl', 'Exterior2nd_Stucco',

                'Functional_Min1', 'Electrical_Mix', 'Exterior1st_WdShing', 'Neighborhood_SWISU', 'GarageQual_Po', 'SaleCondition_AdjLand', 'Electrical_FuseP', 'Functional_Min2',

                'MSZoning_RH', 'BsmtFinType1_ALQ', 'HouseStyle_1Story', 'Exterior1st_HdBoard', 'Heating_Wall', 'GarageCond_Po', 'Exterior1st_BrkComm', 'BsmtFinType1_LwQ', 'FireplaceQu_Po',

                'SaleType_COD', 'GarageType_CarPort', 'BsmtCond_Po', 'LotConfig_Inside', 'RoofMatl_CompShg', 'PavedDrive_P', 'HouseStyle_SFoyer', 'BsmtFinType1_Unf', 'SaleCondition_Normal',

                'Functional_Maj2', 'HouseStyle_1.5Unf', 'BldgType_Twnhs', 'BldgType_2fmCon', 'LandContour_Bnk', 'BldgType_Duplex', 'Neighborhood_Sawyer', 'Condition1_Feedr', 'Neighborhood_BrDale',

                'HeatingQC_Gd', 'Condition1_Artery', 'BsmtFinType1_BLQ', 'Exterior2nd_AsbShng', 'BsmtFinType1_Rec', 'Exterior1st_AsbShng', 'KitchenAbvGr', 'EnclosedPorch', 'Heating_Grav',

                'Neighborhood_MeadowV', 'Foundation_Slab', 'BsmtQual_Fa', 'SaleCondition_Abnorml', 'GarageQual_Fa', 'Electrical_FuseF', 'Neighborhood_NAmes', 'BsmtCond_Fa', 'GarageCond_Fa',

                'Exterior2nd_MetalSd', 'Exterior1st_MetalSd', 'Neighborhood_BrkSide', 'ExterQual_Fa', 'HeatingQC_Fa', 'HouseStyle_1.5Fin', 'RoofStyle_Gable', 'ExterCond_Fa', 'Exterior2nd_Wd Shng',

                'Exterior2nd_Brk Cmn', 'Exterior2nd_Wd Sdng', 'Exterior1st_Wd Sdng', 'MSZoning_C (all)']
# Let's drop the selected variables



unified = unified.drop(list_to_drop, axis = 1)
# Creating a correlation matrix, since we dropped a few columns



corr_matrix = unified.corr()
# Checking the correlations again



corr_matrix['SalePrice'].sort_values(ascending = False)
#unified[~unified.isin([np.nan, np.inf, -np.inf]).any(1)]
# Let's separate X and y again



y = unified['SalePrice']

X = unified.drop('SalePrice', axis = 1)
# Creating the train  test set



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# Creating the Linear Regression and fitting it



lr = LinearRegression()

lr.fit(X_train, y_train)
# Creating the prediction of y



y_pred = lr.predict(X_test)
# Checking the adherence of the model so far



_ = plt.figure(figsize=(8,8))

_ = sns.regplot(x = y_test, y = y_pred)
# Let's calculate the MSE



mse = sklearn.metrics.mean_squared_error(y_test, y_pred)

mse
# Let's calculate the RMSE



rmse = mse**(1/2)

rmse
# Let's calculate the r²



sklearn.metrics.r2_score(y_test, y_pred)
# Getting the X columns as a list



x_columns = list(X.columns)
# Reading the test data



df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df_test.head()
# Let's check the shape of the test DataFrame



df_test.shape
# Let's make the X the same size of the test DataFrame



X = X[:1459]
# Let's create a DataFrame with the ID and the SalePrice in log values



submit = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': lr.predict(X)})
# Let's take a sneak peak of the submit



submit.head()
# Creating the csv file, remember to use index false or it will create 3 columns



submit.to_csv('output.csv', index = False)