import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import sklearn

from sklearn.feature_selection import RFE

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression



import statsmodels

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
## Loading data



geely = pd.read_csv('../input/geely-auto/CarPriceAssignment.csv')

geely.head()
geely.info()



## There are no missing values in any column so no imputation is required
## Check spread of the data to check if there are any outliers

## for most of the variables - mean and median are very close

## Price looks like right skewed



geely.describe()
sns.pairplot(x_vars=['price', 'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight'], y_vars=['price', 'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight'], data=geely)



# Analysis:

# Kind of linear Relationship between (price & carwidth), (wheelbase & carlength), (carlength & carwidth) 
sns.pairplot(x_vars=['price', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio'], y_vars=['price', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio'], data=geely)



# Analysis:

# Kind of linear Relationship between (price & curbweight), (price & enginesize), (curbweight & enginesize) 
sns.pairplot(x_vars=['price', 'horsepower', 'peakrpm', 'citympg', 'highwaympg'], y_vars=['price', 'horsepower', 'peakrpm', 'citympg', 'highwaympg'], data=geely)



# Analysis:

# Kind of linear Relationship between (price & horsepower), (citympg & highwaympg)

# Kind of Non-Linear Relationship between (price & citympg), (price & highwaympg), (horsepower & citympg), (horsepower & highwaympg)
## preprocessing CarName by keeping only car company



def preprocess_carname(x):

    return x.split()[0]



geely.loc[:, 'CarName'] = geely['CarName'].apply(lambda x: preprocess_carname(x))

geely['CarName'].value_counts()
## CarName spelling mistakes correction



def carname_spelling_correction(x):

    if x in ['toyota', 'toyouta']:

        return 'toyota'

    elif x in ['volkswagen', 'vw', 'vokswagen']:

        return 'volkswagen'

    elif x in ['porsche', 'porcshce']:

        return 'porsche'

    elif x in ['nissan', 'Nissan']:

        return 'nissan'

    elif x in ['mazda', 'maxda']:

        return 'mazda'

    elif x in ['alfa-romero']:

        return 'alfa-romeo'

    return x



geely.loc[:, 'CarName'] = geely['CarName'].apply(lambda x: carname_spelling_correction(x))

geely['CarName'].value_counts()
## Plotting car company vs price



plt.figure(figsize=(15, 5))

plt.xticks(rotation=30)

g = sns.boxplot(x='CarName', y='price', data=geely)
## fueltype possible values



geely['fueltype'].value_counts()
## Plotting fuel type vs price



# plt.figure(figsize=(15, 5))

# plt.xticks(rotation=30)

g = sns.boxplot(x='fueltype', y='price', data=geely)
## aspiration possible values



geely['aspiration'].value_counts()
## Plotting aspiration vs price



# plt.figure(figsize=(15, 5))

# plt.xticks(rotation=30)

g = sns.boxplot(x='aspiration', y='price', data=geely)
## doornumber possible values



geely['doornumber'].value_counts()
## Plotting doornumber vs price



# plt.figure(figsize=(15, 5))

# plt.xticks(rotation=30)

g = sns.boxplot(x='doornumber', y='price', data=geely)
## carbody possible values



geely['carbody'].value_counts()
## Plotting carbody vs price



g = sns.boxplot(x='carbody', y='price', data=geely)
## drivewheel possible values



geely['drivewheel'].value_counts()
## Plotting drivewheel vs price



g = sns.boxplot(x='drivewheel', y='price', data=geely)
## enginelocation possible values



geely['enginelocation'].value_counts()
## Plotting enginelocation vs price



g = sns.boxplot(x='enginelocation', y='price', data=geely)
## enginetype possible values



geely['enginetype'].value_counts()
## Plotting enginetype vs price



plt.figure(figsize=(15, 5))

plt.xticks(rotation=30)

g = sns.boxplot(x='enginetype', y='price', data=geely)
## cylindernumber possible values



geely['cylindernumber'].value_counts()
## Plotting cylindernumber vs price



plt.figure(figsize=(15, 5))

plt.xticks(rotation=30)

g = sns.boxplot(x='cylindernumber', y='price', data=geely)
## fuelsystem possible values



geely['fuelsystem'].value_counts()
## Plotting fuelsystem vs price



plt.figure(figsize=(15, 5))

plt.xticks(rotation=30)

g = sns.boxplot(x='fuelsystem', y='price', data=geely)
## dropping car_ID as it is unique for all the data points and independent of the dependent variable (price)



geely = geely.drop(['car_ID'], axis=1)
## convert fuel type, aspiration, door number, engine location to binary 



geely.loc[:, 'fueltype'] = geely['fueltype'].apply(lambda x: 1 if x=='gas' else 0)

geely.loc[:, 'aspiration'] = geely['aspiration'].apply(lambda x: 1 if x=='std' else 0)

geely.loc[:, 'doornumber'] = geely['doornumber'].apply(lambda x: 1 if x=='four' else 0)

geely.loc[:, 'enginelocation'] = geely['enginelocation'].apply(lambda x: 1 if x=='front' else 0)
## create dummy variables all the other categorical variable (there is no order in the labels (except cylindernumber) so label encoder can't be used)



geely = pd.get_dummies(geely, drop_first=True)

geely.head()
geely.info()
## Plotting correlation without dummy categorical variables



plt.figure(figsize=(25, 15))

sns.heatmap(geely[['price', 'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'fueltype', 'aspiration', 'doornumber', 'enginelocation']].corr(), annot=True)



# Analysis

# price is highly correlated with enginesize, curbweight, horsepower, carwidth

# enginesize is highly correlated with curbweight, horsepower, carwidth
## Splitting the data 70-30



geely_train, geely_test = train_test_split(geely, test_size=0.3, random_state=42)

print(geely_train.shape, geely_test.shape)
## Using MinMaxScaler as dummy variables are binary so converting all the other values in the range [0,1]



scaler = MinMaxScaler()



num_vars = ['price', 'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio']



## fitting the Scaler on train data and then transforming train data as per the scaling params

geely_train[num_vars] = scaler.fit_transform(geely_train[num_vars])



## transforming test data as per the scaling params

geely_test[num_vars] = scaler.transform(geely_test[num_vars])
## Train data after scaling



geely_train.head()
## Test data after scaling



geely_test.head()
## Extracting X and y for linear regression modelling



y = geely_train.pop('price')

X = geely_train



print(y.shape, X.shape)
## Creating a LR model object

lr = LinearRegression()



## Creating RFE object

rfe = RFE(lr, 20)



## Fitting RFE (using LR model) and extracting feature importance of all the features

rfe = rfe.fit(X, y)



## listing all the column and corresponding rankings

list(zip(geely_train.columns, rfe.support_, rfe.ranking_))
## Get top 20 most important features



cols = geely_train.columns[rfe.support_]

cols
## Extract data for the top 20 columns



X = geely_train[cols]

X.shape, y.shape
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF'] = [round(variance_inflation_factor(X_train_sm.values, i), 2) for i in range(X_train_sm.shape[1])]

vif.sort_values(by='VIF', ascending='False')
## Dropping cylindernumber_twelve



cols = cols.drop('cylindernumber_twelve')

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by='VIF', ascending='False')
## Dropping CarName_renault



cols = cols.drop('CarName_renault')

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by=['VIF'], ascending='True')
## Dropping cylindernumber_two



cols = cols.drop('cylindernumber_two')

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by=['VIF'], ascending='True')
# Train LR model on train data

lr = LinearRegression()

lr = lr.fit(X, y)

print(lr.coef_)



# Extract test data

y_test = geely_test['price']

X_test = geely_test[cols]



# Compute r2 score for test data

r2_score(y_test, lr.predict(X_test))



## R2 on test data is 86.5% and train data is 96.2% so difference is high so the LR model is not generalised yet
## Dropping enginelocation



cols = cols.drop('enginelocation')

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by=['VIF'], ascending='True')
## Dropping curbweight



cols = cols.drop(['curbweight'])

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by=['VIF'], ascending='True')
## Dropping CarName_peugeot



cols = cols.drop(['CarName_peugeot'])

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by=['VIF'], ascending='True')
## Dropping boreratio



cols = cols.drop(['boreratio'])

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by=['VIF'], ascending='True')
# Train LR model on train data

lr = LinearRegression()

lr = lr.fit(X, y)

print(lr.coef_)



# Extract test data

y_test = geely_test['price']

X_test = geely_test[cols]



# Compute r2 score for test data

r2_score(y_test, lr.predict(X_test))



## R2 on test data is 86.0% and train data is 94.4% so difference is high so the LR model is not generalised yet
## Dropping carwidth



cols = cols.drop(['carwidth'])

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF Factor'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by=['VIF Factor'], ascending='True')
## Dropping CarName_jaguar



cols = cols.drop(['CarName_jaguar'])

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by=['VIF'], ascending='True')
## Dropping carbody_sedan



cols = cols.drop(['carbody_sedan'])

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF Factor'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by=['VIF Factor'], ascending='True')
## Dropping carbody_wagon



cols = cols.drop(['carbody_wagon'])

cols
## Extract data for the 'cols' data columns



X = geely_train[cols]
## Training statsmodels linear regression model



# adding constant to the training data

X_train_sm = sm.add_constant(X)



# running the linear model

lr = sm.OLS(y, X_train_sm)

lr_model = lr.fit()



# sm tuned linear model summary

lr_model.summary()
## Calculate the VIFs of all the independent variables



X_train_sm = X_train_sm.drop(['const'], axis=1)



vif = pd.DataFrame()

vif['features'] = X_train_sm.columns

vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif.sort_values(by=['VIF'], ascending='True')
# Train LR model on train data

lr = LinearRegression()

lr = lr.fit(X, y)

lr.coef_
# Compute residual values for train data

residual = y - lr.predict(X)



# Plot distplot of the residuals

sns.distplot(residual)



## Residuals distribution is Gaussian with 0 mean so for the LR model tuned, all the assumptions are valid
# Extract test data

y_test = geely_test.pop('price')

X_test = geely_test[cols]



# Predict LR model output for test data

y_test_pred = lr.predict(X_test)



# Compute r2 score for test data

r2_score(y_test, y_test_pred)



## R2 on test data is 88.5% and train data is 90.7% so difference is less and thus LR model is generalised
# Plotting y_test and y_test_pred to understand the spread.



# plt.scatter(y_test, y_test_pred)

sns.regplot(y_test, y_test_pred)

plt.title('y test vs y test predicted')

plt.xlabel('y_test', fontsize=18)                          

plt.ylabel('y_pred', fontsize=16)  
## Plotting residuals to check homoscedasticity

## Analysis - No pattern in residuals so seems like homoscadastic



sns.regplot(np.linspace(0, 62, y_test.shape[0]), y_test - y_test_pred)
## Plotting qqplot to match test data residual distribution to standardized normal distribution 

## Analyis test data residuals are indeed normally distributed except few points 



sm.qqplot(y_test - y_test_pred, line='s')

plt.show()