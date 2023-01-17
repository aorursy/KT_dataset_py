import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing the required libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.model_selection import cross_val_score



# hide warnings

import warnings

warnings.filterwarnings('ignore')



#To print all columns

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows',100)
# Reading the csv into a dataframe

# From the data dictionary, it is seen that some categorical variables has 'NA' as an option.

# It cannot be considered as null as it holds some meaning here.

# Thus reading the data with the parameter keep_default_na=False such that it will not consider NA as null



df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", keep_default_na=False)
# Looking at the data

df.head(10)
# Looking at the columns and its data types

df.info()
df.select_dtypes('object').head()
# We can see from the above info that 'LotFrontage', 'MasVnrArea' and 'GarageYrBlt' are considered as 'object' 

# inspite being numerical cols

# It means they have NA values in the columns



# Replacing with 0 because the houses who do not have any street connected to property will get a 'NA', which basically means 0ft

df['LotFrontage']=df['LotFrontage'].replace('NA',0) 



# Replacing with 0 because for the houses where Masonry veneer type is none, the area covered will obviously be 0

df['MasVnrArea']=df['MasVnrArea'].replace('NA',0)



# If the house do not have any garage, the yr built does not make sense, so keeping it as 0

df['GarageYrBlt']=df['GarageYrBlt'].replace('NA',0)
# Converting these columns back to numeric

df['LotFrontage']=df['LotFrontage'].astype('int64')

df['MasVnrArea']=df['MasVnrArea'].astype('int64')

df['GarageYrBlt']=df['GarageYrBlt'].astype('int64')



# Some columns which were considered as numeric are actually categorial. 

# Converting them to object

df['MSSubClass']=df['MSSubClass'].astype('object')

df['OverallQual']=df['OverallQual'].astype('object')

df['OverallCond']=df['OverallCond'].astype('object')
# Checking if the ID column has duplicated values

sum(df['Id'].duplicated())
# As this column is unique, we can set it as index

df.set_index('Id', inplace=True)
# Looking at the final number of rows and columns

df.shape
# Creating new columns out of the Year columns

df['YearsOldWhenSold']=df['YrSold']-df['YearBuilt']

df['WasRemodelled']=df.apply(lambda x: 0 if (x['YearBuilt']==x['YearRemodAdd']) else 1, axis=1)

df['WasGarageAddedLater'] = df.apply(lambda x: 1 if (x['GarageYrBlt'] > x['YearBuilt']) else 0, axis=1)



# Converting the newly created columns to object datatype from int datatype

df['WasRemodelled']=df['WasRemodelled'].astype('object')

df['WasGarageAddedLater']=df['WasGarageAddedLater'].astype('object')
# Droping the columns with Year values as we do not need them anymore

df.drop(columns=['YearBuilt','GarageYrBlt','MoSold','YrSold','YearRemodAdd'],inplace=True)
# Looking at the data

df.head()
# Creating two lists of numeric and categorical columns

numericCols=df.select_dtypes(exclude='object').columns

categoricalCols=df.select_dtypes(include='object').columns
# Stats of Numeric columns

df[numericCols].describe().T
# Looking at the value counts of all the categorical columns

for col in categoricalCols:

    print("Value Counts for category " + col)

    print(df[col].value_counts())

    print("--------------------")
# We can see that 'Electrical' has NA value, but in data dictionary, NA is not listed as a suitable value for Electrical

# So, we remove that row considering it to be a null value

df.drop(index=df[df['Electrical']=='NA'].index,inplace=True)
# From the above graph we can see that the following columns donot show any variability in data, thus dropping them

df.drop(columns=['PoolQC','Utilities','Condition2','RoofMatl'],inplace=True)
# We have identified the following columns as having outliers by looking at the above plots

colsWithOutliers = ['LotFrontage','LotArea','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','1stFlrSF','GrLivArea','GarageArea']



# For all these columns, removing the outliers by using the IQR method

for col in colsWithOutliers:

    Q1 = df[col].quantile(0.05)

    Q3 = df[col].quantile(0.95)

    IQR = Q3 - Q1

    df = df[((df[col] >= (Q1 - 1.5 * IQR))& (df[col] <= (Q3 + 1.5 * IQR)))]
# Looking at the shape of the df to determine the number of rows dropped due to outlier treatment

df.shape
# Calculating the skewness of the SalePrice variable

df['SalePrice'].skew()
# We see that the dependent variable 'SalePrice' is skewed and tailed, 

# so to make it a normal distribution, we need to log transform it

df['SalePrice'] = np.log(df['SalePrice'])
# Calculating the skewness of the SalePrice variable

df['SalePrice'].skew()
df.drop(columns=['Exterior1st','MSSubClass','MSZoning','Electrical','Heating','HouseStyle','MiscFeature'],inplace=True)
# Creating a list of numeric variables which we will scale later

colsToScale=df.select_dtypes(exclude='object').columns.tolist()

colsToScale.remove('SalePrice')

colsToScale
# These columns already have label encoded, so just converting the data types to numerical

df['OverallQual']=df['OverallQual'].astype('int64')

df['OverallCond']=df['OverallCond'].astype('int64')

df['WasRemodelled']=df['WasRemodelled'].astype('int64')

df['WasGarageAddedLater']=df['WasGarageAddedLater'].astype('int64')
newExValues = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

df['ExterQual']=df['ExterQual'].map(newExValues)

df['ExterCond']=df['ExterCond'].map(newExValues)

df['HeatingQC']=df['HeatingQC'].map(newExValues)

df['KitchenQual']=df['KitchenQual'].map(newExValues)
newBsValues = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

df['BsmtQual']=df['BsmtQual'].map(newBsValues)

df['BsmtCond']=df['BsmtCond'].map(newBsValues)

df['FireplaceQu']=df['FireplaceQu'].map(newBsValues)

df['GarageQual']=df['GarageQual'].map(newBsValues)

df['GarageCond']=df['GarageCond'].map(newBsValues)
newBsExValues={'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

df['BsmtExposure']=df['BsmtExposure'].map(newBsExValues)
newBsmtFValues = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ' : 6}

df['BsmtFinType1']=df['BsmtFinType1'].map(newBsmtFValues)

df['BsmtFinType2']=df['BsmtFinType2'].map(newBsmtFValues)
# Creating dummies for categorical columns

categoricalCols=df.select_dtypes(include='object').columns

dummies = pd.get_dummies(df[categoricalCols], drop_first = True)
# Merging the dummies to the dataframe

df = pd.concat([df, dummies], axis = 1)



# Dropping the actual categorical columns

df.drop(columns=categoricalCols,inplace=True)



df.shape
#Putting independent variables to X

X = df.drop(['SalePrice'], axis=1)

X.head()
X.shape
#Putting response variable to y

y = df['SalePrice']

y.head()
#Splitting the data into test and train set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
# Scaling all the numeric columns

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[colsToScale] = scaler.fit_transform(X_train[colsToScale])

X_train.head()
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 20, 50, 100, 500, 1000 ]}





ridge = Ridge()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train, y_train) 
# Finding the optimal value of alpha

model_cv.best_params_
# Best score : 

model_cv.best_score_
# Looking at the results of gridsearch cv

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=200]

cv_results.head()
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
# Considering optimal value of alpha as 13 and fitting the Ridge regression

alpha = 13

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
# Creating a list of all the variables and their coefficients

ridgeCoeff=list(zip(ridge.coef_, X_train.columns))
# Plotting 10 variables which have +ve correlation with price, and 10 variables which have -ve correlation with price

ridgeCoefDf=pd.DataFrame(ridgeCoeff).sort_values(by=0)

ridgeCoefDf10=pd.concat([ridgeCoefDf.iloc[:10],ridgeCoefDf.iloc[-10:]],axis=0)



plt.figure(figsize=(12,12))

sns.barplot(x=ridgeCoefDf10[0],y=ridgeCoefDf10[1])

plt.xlabel("Coefficients")

plt.ylabel("Variables")

plt.title("Variables-Coefficients graph")
# Looking at the top 15 variables having the highest coefficients

nonZeroRidgeCoeff=sorted([x for x in ridgeCoeff if abs(x[0])!=0], key=lambda x: abs(x[0]),reverse=True)

nonZeroRidgeCoeff[:15]
# Number of variables having non-zero coefficients

len(nonZeroRidgeCoeff)
# Total number of columns

len(X_train.columns)
# Predicting the price

y_pred_ridge = ridge.predict(X_train)
# Calculating the r2 score

metrics.r2_score(y_pred=y_pred_ridge,y_true=y_train)
# Calculating the mean squared error of the model

np.sqrt(metrics.mean_squared_error(y_true=y_train, y_pred=y_pred_ridge))
# Calculating the Average root mean squared error

scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

np.mean(np.sqrt(-scores))
# Calculating the Average r2 score

scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')

np.mean(scores)
# Plot the histogram of the error terms

fig = plt.figure()

residuals=y_train - y_pred_ridge

sns.distplot(residuals, bins = 20)

fig.suptitle('Error Terms') 

plt.xlabel('Errors')  



plt.plot([0, 0], [0, 5])
# Calculating the number of data points

y_pred_ridge.shape
# Plot showing how the errors are distributed

fig = plt.figure()

c = [i for i in range(1,1004,1)]

plt.scatter(c,y_train-y_pred_ridge)



fig.suptitle('Error Terms')

plt.xlabel('Index')

plt.ylabel('ytrain-ytrain_price')
# Scaling the numeric columns in test dataframe

X_test[colsToScale] = scaler.transform(X_test[colsToScale])

X_test.head()
# Making predictions on test dataset

y_pred_test_ridge = ridge.predict(X_test)
# Calculating at the number of data points of test set

y_pred_test_ridge.shape
# Plot showing how the errors are distributed

fig = plt.figure()

c = [i for i in range(1,432,1)] # the test dataset has 63 values

plt.scatter(c,y_test-y_pred_test_ridge)



fig.suptitle('Error Terms')

plt.xlabel('Index')

plt.ylabel('ytest-ypred')
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

sns.regplot(y_test,y_pred_test_ridge)

fig.suptitle('y_test vs y_pred')

plt.xlabel('y_test')

plt.ylabel('y_pred')
# Calculating the r2 score

metrics.r2_score(y_pred=y_pred_test_ridge, y_true=y_test)
# Calculating the mean squared error of the model

np.sqrt(metrics.mean_squared_error(y_pred=y_pred_test_ridge, y_true=y_test))
# Calculating the Average root mean squared error

scores = cross_val_score(ridge, X_test, y_test, cv=5, scoring='neg_mean_squared_error')

np.mean(np.sqrt(-scores))
# Calculating the Average r2 score

scores = cross_val_score(ridge, X_test, y_test, cv=5, scoring='r2')

np.mean(scores)
params = {'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0 ]}

lasso = Lasso()



# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train, y_train) 
# Looking at the optimal value of alpha

model_cv.best_params_
# Best score : 

model_cv.best_score_
# Looking at the results of grid search cv

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=0.001]

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
# Fitting the lasso regression for the optimal value of alpha

alpha =0.0005



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 
lasso.coef_
# Creating a list of all the variables and their coefficients

lassoCoeff=list(zip(lasso.coef_, X_train.columns))
# Plotting 10 variables which have +ve correlation with price, and 10 variables which have -ve correlation with price

lassoCoefDf=pd.DataFrame(lassoCoeff).sort_values(by=0)

lassoCoefDf10=pd.concat([lassoCoefDf.iloc[:10],lassoCoefDf.iloc[-10:]],axis=0)



plt.figure(figsize=(12,12))

sns.barplot(x=lassoCoefDf10[0],y=lassoCoefDf10[1])

plt.xlabel("Coefficients")

plt.ylabel("Variables")

plt.title("Variables-Coefficients graph")
# Looking at the top 15 variables having the highest coefficients

nonZeroLassoCoeff=sorted([x for x in lassoCoeff if abs(x[0])!=0], key=lambda x: abs(x[0]),reverse=True)

nonZeroLassoCoeff[:15]
# Number of variables having non zero coefficients

len(nonZeroLassoCoeff)
# Printing the variables in the order of importance

[x[1] for x in nonZeroLassoCoeff]
# Predicting the price

y_pred_lasso = lasso.predict(X_train)
# Calculating the r2 score

metrics.r2_score(y_pred=y_pred_lasso,y_true=y_train)
# Calculating the mean squared error of the model

np.sqrt(metrics.mean_squared_error(y_true=y_train, y_pred=y_pred_lasso))
# Calculating the Average root mean squared error

scores = cross_val_score(lasso, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

np.mean(np.sqrt(-scores))
# Calculating the Average r2 score

scores = cross_val_score(lasso, X_train, y_train, cv=5, scoring='r2')

np.mean(scores)
# Plot the histogram of the error terms

fig = plt.figure()

residuals=y_train - y_pred_lasso

sns.distplot(residuals, bins = 20)

fig.suptitle('Error Terms') 

plt.xlabel('Errors')  



plt.plot([0, 0], [0, 5])
# Looking at the number of data points

y_pred_lasso.shape
# Plot showing how the errors are distributed

fig = plt.figure()

c = [i for i in range(1,1004,1)]

plt.scatter(c,y_train-y_pred_lasso)



fig.suptitle('Error Terms')

plt.xlabel('Index')

plt.ylabel('ytrain-ytrain_price')
# Making predictions on test dataset

y_pred_test_lasso = lasso.predict(X_test)
# Looking at the number of data points of test set

y_pred_test_lasso.shape
# Plot showing how the errors are distributed

fig = plt.figure()

c = [i for i in range(1,432,1)] # the test dataset has 63 values

plt.scatter(c,y_test-y_pred_test_lasso)



fig.suptitle('Error Terms')

plt.xlabel('Index')

plt.ylabel('ytest-ypred')
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

sns.regplot(y_test,y_pred_test_lasso)

fig.suptitle('y_test vs y_pred')

plt.xlabel('y_test')

plt.ylabel('y_pred')
# Calculating the r2 score

metrics.r2_score(y_pred=y_pred_test_lasso,y_true=y_test)
# Calculating the mean squared error of the model

np.sqrt(metrics.mean_squared_error(y_true=y_test, y_pred=y_pred_test_lasso))
# Calculating the Average root mean squared error

scores = cross_val_score(lasso, X_test, y_test, cv=5, scoring='neg_mean_squared_error')

np.mean(np.sqrt(-scores))
# Calculating the Average r2 score

scores = cross_val_score(lasso, X_test, y_test, cv=5, scoring='r2')

np.mean(scores)
tdf=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", keep_default_na=False)
tdf.info()
def tdfShape():

    print(tdf.shape)

    

tdfShape()
# Replacing with 0 because the houses who do not have any street connected to property will get a 'NA', which basically means 0ft

tdf['LotFrontage']=tdf['LotFrontage'].replace('NA',0) 



# Replacing with 0 because for the houses where Masonry veneer type is none, the area covered will obviously be 0

tdf['MasVnrArea']=tdf['MasVnrArea'].replace('NA',0)



# If the house do not have any garage, the yr built does not make sense, so keeping it as 0

tdf['GarageYrBlt']=tdf['GarageYrBlt'].replace('NA',0)



tdf['Functional']=tdf['Functional'].replace('NA',tdf['Functional'].mode()[0])



tdf['SaleType']=tdf['SaleType'].replace('NA',tdf['SaleType'].mode()[0])
tdfShape()
tdf.drop(columns=['Exterior1st','MSSubClass','MSZoning','Electrical','Heating','HouseStyle','MiscFeature'],inplace=True)
# Converting these columns back to numeric

tdf['LotFrontage']=tdf['LotFrontage'].astype('int64')

tdf['MasVnrArea']=tdf['MasVnrArea'].astype('int64')

tdf['GarageYrBlt']=tdf['GarageYrBlt'].astype('int64')



# Some columns which were considered as numeric are actually categorial. 

# Converting them to object

tdf['OverallQual']=tdf['OverallQual'].astype('object')

tdf['OverallCond']=tdf['OverallCond'].astype('object')
# Creating new columns out of the Year columns

tdf['YearsOldWhenSold']=tdf['YrSold']-tdf['YearBuilt']

tdf['WasRemodelled']=tdf.apply(lambda x: 0 if (x['YearBuilt']==x['YearRemodAdd']) else 1, axis=1)

tdf['WasGarageAddedLater'] = tdf.apply(lambda x: 1 if (x['GarageYrBlt'] > x['YearBuilt']) else 0, axis=1)



# Converting the newly created columns to object datatype from int datatype

tdf['WasRemodelled']=tdf['WasRemodelled'].astype('object')

tdf['WasGarageAddedLater']=tdf['WasGarageAddedLater'].astype('object')
# Droping the columns with Year values as we do not need them anymore

tdf.drop(columns=['YearBuilt','GarageYrBlt','MoSold','YrSold','YearRemodAdd'],inplace=True)
tdf.drop(columns=['PoolQC','Utilities','Condition2','RoofMatl'],inplace=True)
# These columns already have label encoded, so just converting the data types to numerical

tdf['OverallQual']=tdf['OverallQual'].astype('int64')

tdf['OverallCond']=tdf['OverallCond'].astype('int64')

tdf['WasRemodelled']=tdf['WasRemodelled'].astype('int64')

tdf['WasGarageAddedLater']=tdf['WasGarageAddedLater'].astype('int64')
newExValues = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

tdf['ExterQual']=tdf['ExterQual'].map(newExValues)

tdf['ExterCond']=tdf['ExterCond'].map(newExValues)

tdf['HeatingQC']=tdf['HeatingQC'].map(newExValues)

tdf['KitchenQual']=tdf['KitchenQual'].map(newExValues)
newBsValues = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

tdf['BsmtQual']=tdf['BsmtQual'].map(newBsValues)

tdf['BsmtCond']=tdf['BsmtCond'].map(newBsValues)

tdf['FireplaceQu']=tdf['FireplaceQu'].map(newBsValues)

tdf['GarageQual']=tdf['GarageQual'].map(newBsValues)

tdf['GarageCond']=tdf['GarageCond'].map(newBsValues)
newBsExValues={'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

tdf['BsmtExposure']=tdf['BsmtExposure'].map(newBsExValues)
newBsmtFValues = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ' : 6}

tdf['BsmtFinType1']=tdf['BsmtFinType1'].map(newBsmtFValues)

tdf['BsmtFinType2']=tdf['BsmtFinType2'].map(newBsmtFValues)
tdfShape()
dummies = pd.get_dummies(tdf[categoricalCols], drop_first = True)

tdf = pd.concat([tdf, dummies], axis = 1)



# Dropping the actual categorical columns

tdf.drop(columns=categoricalCols,inplace=True)
tdf.shape
for col in colsToScale:

    tdf[col].replace('NA',0,inplace=True)

    tdf[col].replace(np.nan,0,inplace=True)

    tdf[col]=tdf[col].astype('int64')
tdf[colsToScale] = scaler.transform(tdf[colsToScale])
tdf.set_index('Id',inplace=True)
tdf['KitchenQual'].replace(np.nan,1.0,inplace=True)
tdf_pred=lasso.predict(tdf)
tdf.reset_index(inplace=True)
tdf.head()
finalDf=pd.DataFrame(tdf['Id'])

finalDf['SalePrice']=np.exp(tdf_pred)
finalDf.to_csv("output.csv",index=False)