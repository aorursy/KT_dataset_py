# Importing all required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Importing carprice.csv

car_price_df = pd.read_csv("../input/data.csv")
# Getting categorical variable and splitting the and choosing only one of them to reduce the number of columns after converting them to numerical

car_price_df['Market Category'] = car_price_df['Market Category'].str.split(',', expand = True)[0]

car_price_df['Engine Fuel Type'] = car_price_df['Engine Fuel Type'].str.split(' ', expand = True)[0]

car_price_df['Driven_Wheels'] = car_price_df['Driven_Wheels'].str.split(' ', expand = True)[0]

car_price_df['Vehicle Style'] = car_price_df['Vehicle Style'].str.split(' ', expand = True)[0]

# Model is not giving any value addition to prediction so dropping it

car_price_df.drop('Model', inplace=True,axis=1)
# Check the head of the dataset

car_price_df.head()
# checking shape of the dataframe

car_price_df.shape
# checking values for null

car_price_df.info()
# removing all the rows from DF which has null values

CarPriceCleanDf=car_price_df.dropna()
# checking null values after removing them

CarPriceCleanDf.info()
# getting all the important details/information of dataset

CarPriceCleanDf.describe()
# Pair plot to check which feature has strongest relationship with 'MSRP' to get the benchmark model i.e. Simple Linear Regression model

sns.pairplot(CarPriceCleanDf, x_vars=['Engine HP', 'city mpg', 'Vehicle Size'], y_vars='MSRP',size=4, aspect=1, kind='scatter')

plt.show()
fig = plt.figure(figsize=(12,10))

sns.heatmap(CarPriceCleanDf.corr(),annot=True)
# converting all categorical variables to numeric variables



# Let's drop the first column from status df using 'drop_first = True'

carmake = pd.get_dummies(CarPriceCleanDf['Make'], drop_first = True)



carenginefueltype = pd.get_dummies(CarPriceCleanDf['Engine Fuel Type'], drop_first = True)



cartransmissiontype = pd.get_dummies(CarPriceCleanDf['Transmission Type'], drop_first = True)



cardrivenwheels = pd.get_dummies(CarPriceCleanDf['Driven_Wheels'], drop_first = True)



carmarketcategory =  pd.get_dummies(CarPriceCleanDf['Market Category'], drop_first = True)



carvehiclesize = pd.get_dummies(CarPriceCleanDf['Vehicle Size'], drop_first = True)



carvehiclestyle = pd.get_dummies(CarPriceCleanDf['Vehicle Style'], drop_first = True)



# Add the results to the original housing dataframe

CarPriceCleanDf = pd.concat([CarPriceCleanDf,carmake, carenginefueltype,cartransmissiontype,cardrivenwheels,

                             carmarketcategory,carvehiclesize,carvehiclestyle], axis = 1)



# Drop all categorical variable original columns

CarPriceCleanDf.drop(['Make','Engine Fuel Type','Transmission Type','Driven_Wheels', 'Market Category','Vehicle Size','Vehicle Style',], axis = 1, inplace = True)



# Now let's see the head of our dataframe.

CarPriceCleanDf.head()
# import train_test_split

from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)



# splitting train and test dataset for model

df_train, df_test = train_test_split(CarPriceCleanDf, train_size = 0.7, test_size = 0.3, random_state = 100)
# import MinMaxScaler to scale all umerical features(other than 1 and 0)

from sklearn.preprocessing import MinMaxScaler
# creating object of MinMaxScaler

scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'city mpg','Popularity','MSRP']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



df_train.head()
# creating X and y from training dataset



y_train = df_train.pop('MSRP')

X_train = df_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 15

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 15)             # running RFE

rfe = rfe.fit(X_train, y_train)
# Getiing the rank of features and support value



list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# select the feature with 'True' support value

col = X_train.columns[rfe.support_]

col
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable to get intercept

import statsmodels.api as sm  

X_train_rfe_lm = sm.add_constant(X_train_rfe)
# fitting the model

lm = sm.OLS(y_train,X_train_rfe_lm).fit()  
#Let's see the summary of our linear model with multiple features

print(lm.summary())
# Dropping the 'UNKNOWN' column from selected features

X_train_new1 = X_train_rfe.drop(["UNKNOWN"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm1 = sm.add_constant(X_train_new1)



# creating new model after removing 'UNKNOWN' column

lm = sm.OLS(y_train,X_train_lm1).fit()   



#Let's see the summary of new linear model

print(lm.summary())
# Dropping Spyker from the selected features

X_train_new2 = X_train_new1.drop(["Spyker"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm2 = sm.add_constant(X_train_new2)

lm = sm.OLS(y_train,X_train_lm2).fit()   



#Let's see the summary of our linear model

print(lm.summary())
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new2.columns

vif['VIF'] = [variance_inflation_factor(X_train_new2.values, i) for i in range(X_train_new2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# fitting model to predict the Y-values of training dataset

y_train_price = lm.predict(X_train_lm2)
# Error term should be normally distributed along mean zero for a stable model



fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)     
df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('MSRP')

X_test = df_test
# Now let's use our model to make predictions.



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new2 = X_test[X_train_new2.columns]



# Adding a constant variable 

X_test_new_lm2 = sm.add_constant(X_test_new2)
# Making predictions

y_pred = lm.predict(X_test_new_lm2)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)  
# import mean squared error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
#Returns the mean squared error; we'll take a square root

np.sqrt(mean_squared_error(y_test, y_pred))
# getting R-squared value

r_squared = r2_score(y_test, y_pred)

r_squared
# creating X and y from training dataset

X = CarPriceCleanDf['Engine HP']

y = CarPriceCleanDf['MSRP']
# import train_test_split

from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)



# splitting train and test dataset for model



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
# Add a constant to get an intercept

X_train_sm = sm.add_constant(X_train)



# Fit the resgression line using 'OLS'

lr = sm.OLS(y_train, X_train_sm).fit()



# get the model summary

print(lr.summary())
# fitting model to predict the training value and get the residual



y_train_pred = lr.predict(X_train_sm)

res = (y_train - y_train_pred)
# Error term should be normally distributed along mean zero for a stable model



fig = plt.figure()

sns.distplot(res, bins = 15)

fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 

plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label

plt.show()
# Drawing scatter plot of residuals



plt.scatter(X_train,res)

plt.show()
# Add a constant to X_test

X_test_sm = sm.add_constant(X_test)



# Predict the y values corresponding to X_test_sm

y_pred = lr.predict(X_test_sm)
#Returns the mean squared error; we'll take a square root

np.sqrt(mean_squared_error(y_test, y_pred))
# get R-squared value

r_squared = r2_score(y_test, y_pred)

r_squared