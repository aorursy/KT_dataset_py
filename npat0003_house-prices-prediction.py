# Packages to perform dataframe manipulations, statistical calculations and plotting

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression, RidgeCV

from sklearn.metrics import mean_squared_error, make_scorer
# Read the csv into a pandas dataframe

houses = pd.read_csv('../input/train.csv')

#houses2 = pd.read_csv('../input/test.csv')
# Lets merge the two dataframes so we can work on it

#houses = pd.merge(houses1, houses2, how='outer')
print (houses.head())
# Gives an idea about number of rows, columns, data types and also null values

houses.info()
# Lets check the Alley column 

print (houses.Alley.value_counts())

print (houses.Alley.unique())
# Lets replace the NaN with NA (No alley access)

houses['Alley'] = houses.Alley.fillna('NA')

# Lets check now after imputting

print (houses.Alley.value_counts())
# Lets start with the PoolQC

print (houses.PoolQC.value_counts())

print (houses.PoolQC.unique())
# Lets check how many houses actually have pools 

print (houses[houses.PoolArea > 0])

# 10 houses already have PoolQC 3 rows are missing, Let's impute the missing rows with one of the values

houses.loc[(houses.PoolArea>0) & (houses.PoolQC.isnull()==True), 'PoolQC'] = houses['PoolQC'].fillna(houses['PoolQC'].value_counts().index[0])

# Imputting the null values with NA

# Find all the rows without the pool and replace with 'NA'

houses['PoolQC'] = houses['PoolQC'].fillna('NA')

print (houses.PoolQC.value_counts())
print (houses.Fence.value_counts())

print (houses.Fence.unique())

houses['Fence'] = houses.Fence.fillna('NA')

print (houses.Fence.value_counts())
# Lets check the houses with no MasVnr

print (houses.MasVnrArea.value_counts())
print (houses.MasVnrType.value_counts())
# MasVnr area null value replace with 0

houses['MasVnrArea'] = houses.MasVnrArea.fillna(0.0)

## MasVnrType where MasVnrArea >0 and MasVnrType is null

houses.loc[(houses.MasVnrArea>0) & (houses.MasVnrType.isnull()==True), 'MasVnrType']= houses['MasVnrType'].fillna(houses['MasVnrType'].value_counts().index[0])



# Lets impute the null with None for MasVnrType 

houses['MasVnrType']=houses.MasVnrType.fillna('None')

print (houses.MasVnrType.value_counts())
print (houses.TotalBsmtSF.value_counts())
# There are 72 houses that have no basement, lets impute NA for null there

houses.loc[houses.TotalBsmtSF == 0.0, ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] ='NA'

# For TotalBsmtSF > 0 , impute the values by the most repeated ones

col_bsmt_list = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for col in col_bsmt_list:

    houses[col] = houses[col].fillna(houses[col].value_counts().index[0])

houses.loc[houses.BsmtUnfSF.isnull()==True, 'BsmtUnfSF']=houses.BsmtUnfSF.median()
# Lets find the houses with fireplace

print (houses.Fireplaces.value_counts())
# There are 1420 houses with no fireplace, so those houses will have NA for fireplace quality

houses.loc[houses.Fireplaces == 0, 'FireplaceQu'] = 'NA'
# If the GarageArea is not greater then 0, then there is no garage in the house

houses.GarageArea.value_counts()
# There are 157 houses without garage, lets fix that first.

houses.loc[houses.GarageArea==0, ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']]='NA'
# If MiscVal is 0, then there is no MiscFeature



print (houses.MiscVal.value_counts())
# Impute NA where MiscVal is 0

houses.loc[houses.MiscVal==0, 'MiscFeature']='NA'
# Lets replace the null with median house lot frontage

houses['LotFrontage'] = houses.LotFrontage.fillna(houses['LotFrontage'].median())
# If you want to fill every categorical column with its own most frequent value we can use

cat_cols = houses.select_dtypes(include = ['O']).columns.values

print (cat_cols)

cat_col_list = list(cat_cols)

for col in cat_col_list:

    houses[col] = houses[col].fillna(houses[col].value_counts().index[0])
# If you want to fill every numerical column with missing values with its own most frequent value we can use

num_cols = houses.select_dtypes(include = ['float64']).columns.values

for col in num_cols:

    houses[col] = houses[col].fillna(houses[col].median())
houses.info()
houses.shape
# Correlation for numeric columns

corr_houses = houses.corr()

print ("\nThe correlation with respect to house price\n")

print (corr_houses['SalePrice'])

names = corr_houses.columns

plt.figure(figsize=(12, 12))

sns.heatmap(corr_houses, vmax=1, square=True)

plt.show()
# Lets transform the categorical to numeric for model building

le = LabelEncoder()

cat_cols = houses.select_dtypes(include = ['O']).columns.values

print (cat_cols)

cat_col_list = list(cat_cols)

for col in cat_col_list:

    houses[col] = le.fit_transform(houses[col])
feature_names = houses.ix[:, (houses.columns != ['Id']) & (houses.columns != ['SalePrice'])]

feature_names = feature_names.columns

x = houses[feature_names]

y = houses['SalePrice']



# Partition the dataset in train + validation sets

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

print("X_train : " + str(X_train.shape))

print("X_test : " + str(X_test.shape))

print("y_train : " + str(y_train.shape))

print("y_test : " + str(y_test.shape))

# Lets have train and test data

#train = houses[houses['SalePrice']!=0]

#test = houses[houses['SalePrice']==0]
numerical_features = x.select_dtypes(exclude = ["object"]).columns
# Standardize numerical features

stdSc = StandardScaler()

X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])

X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])
# Define error measure for official scoring : RMSE

#(https://www.kaggle.com/juliencs/house-prices-advanced-regression-techniques/a-study-on-regression-applied-to-the-ames-dataset)

scorer = make_scorer(mean_squared_error, greater_is_better = False)



def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))

    return(rmse)



def rmse_cv_test(model):

    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))

    return(rmse)
# Linear Regression

regr = LinearRegression()

regr.fit(X_train, y_train)

predict_price = regr.predict(X_test)

predict_price = np.round(predict_price,decimals=2)

#print (predict_price)



# This part of regression code has been taken from juliencs, thanks.

#(https://www.kaggle.com/juliencs/house-prices-advanced-regression-techniques/a-study-on-regression-applied-to-the-ames-dataset)



# Look at predictions on training and validation set

print("RMSE on Training set :", rmse_cv_train(regr).mean())

print("RMSE on Test set :", rmse_cv_test(regr).mean())



y_train_pred = regr.predict(X_train)

y_test_pred = regr.predict(X_test)



# Plot residuals

plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()
# 2* Ridge

ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 

                cv = 10)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)



print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())

print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())

y_train_rdg = ridge.predict(X_train)

y_test_rdg = ridge.predict(X_test)



# Plot residuals

plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_rdg, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_rdg, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()



# Plot important coefficients

coefs = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(15),

                     coefs.sort_values().tail(15)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Ridge Model")

plt.show()
# fit an Extra Trees model to the data

model = ExtraTreesClassifier()

model.fit(X_train, y_train)

# display the relative importance of each attribute

importance = model.feature_importances_

print (importance)
# model is of type array, convert to type dataframe



imp = pd.DataFrame({'feature':X_train.columns,'importance':np.round(model.feature_importances_,3)})

imp = imp.sort_values('importance',ascending=False).set_index('feature')

print (imp)

imp.plot.bar()

plt.show()