# ignore the warnings

import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#importing the data

path = '/kaggle/input/geely-auto/CarPriceAssignment.csv'

carData = pd.read_csv( path )
carData.head( )
#inspecting the various details related to the car

carData.shape
carData.describe( )
numericCols = [ i for i in carData.columns if carData[ i ].dtype != 'object']

#as car_id is unique it is not required for plotting purposes.

numericCols = list( set( numericCols ) - set( [ 'car_ID' ] ) )

categoricalCols = [ i for i in carData.columns if carData[ i ].dtype == 'object']
numericCols
plt.figure( figsize=( 50, 35 ) )

sns.pairplot( carData[ numericCols ] )

plt.show()
plt.figure(figsize=(30, 25 ))

#sns.set( font_scale = 1.6 )



plt.subplot(3,4,1)

ax = sns.boxplot(x = 'fueltype', y = 'price', data = carData)

ax.set_xticklabels(ax.get_xticklabels(),rotation= 90 )



plt.subplot(3,4,2)

ax = sns.boxplot(x = 'aspiration', y = 'price', data = carData)

ax.set_xticklabels(ax.get_xticklabels(),rotation= 90 )



plt.subplot(3,4,3)

ax = sns.boxplot(x = 'doornumber', y = 'price', data = carData)

ax.set_xticklabels(ax.get_xticklabels(),rotation= 90 )



plt.subplot(3,4,4)

ax = sns.boxplot(x = 'carbody', y = 'price', data = carData )

ax.set_xticklabels(ax.get_xticklabels(),rotation= 90 )



plt.subplot(3,4,5)

ax = sns.boxplot(x = 'drivewheel', y = 'price', data = carData)

ax.set_xticklabels(ax.get_xticklabels(),rotation= 90 )



plt.subplot(3,4,6)

ax = sns.boxplot(x = 'enginelocation', y = 'price', data = carData)

ax.set_xticklabels(ax.get_xticklabels(),rotation= 90 )



plt.subplot(3,4,7)

ax = sns.boxplot(x = 'enginetype', y = 'price', data = carData )

ax.set_xticklabels(ax.get_xticklabels(),rotation= 90 )



plt.subplot(3,4,8)

ax = sns.boxplot(x = 'cylindernumber', y = 'price', data = carData )

ax.set_xticklabels(ax.get_xticklabels(),rotation= 90 )



plt.subplot(3,4,9 )

ax = sns.boxplot(x = 'fuelsystem', y = 'price', data = carData )

ax.set_xticklabels(ax.get_xticklabels(),rotation= 90 )







plt.show()
# List of variables to map

varlist =  ['doornumber', 'cylindernumber' ]



# Defining the map function

def binary_map(x):

    return x.map({ "two": 2, "three" : 3, "four" : 4, "five" : 5, "six" : 6, "eight": 8, "twelve" : 12 })



# Applying the function to the carData list

carData[varlist] = carData[varlist].apply(binary_map)
carData.head( )
#CarName has space between the company and model, thus we split the column on basis of 'space'

data = carData['CarName'].str.split( " ", n = 1, expand = True)

carData[ 'carName' ] = data[ 0 ]

#Dropping the original 'CarName' column

carData.drop( columns =[ "CarName" ], inplace = True ) 
carData.head( )
#Looking at the unique values of the carName, to ensure that there are no inconsistences 

carData[ 'carName' ].unique( )
#Based on the unique values, we can see that there are mistakes in the carName column and we should correct it 

#mazda, toyota, porsche, volkswagen are incorrectly spelled

car_map = { "porcshce": "porsche", "vokswagen" : "volkswagen", "toyouta" : "toyota", "maxda" : "mazda" }

# Applying the function to the carData list

carData[ 'carName' ].replace( car_map, inplace = True ) 

# converting all the names to lowercase

carData[ 'carName' ] = carData[ 'carName' ].str.lower( )
# since fuelType can be identified with 1 column only, we are selecting only 1 column

gas = pd.get_dummies( carData[ 'fueltype' ], drop_first = True)
body = pd.get_dummies( carData[ 'carbody' ], drop_first = True )
engine = pd.get_dummies( carData[ 'enginetype' ], drop_first = True )
wheels = pd.get_dummies( carData[ 'drivewheel' ], drop_first = True )
asp = pd.get_dummies( carData[ 'aspiration' ], drop_first = True )
location = pd.get_dummies( carData[ 'enginelocation' ], drop_first = True )
system = pd.get_dummies( carData[ 'fuelsystem' ], drop_first = True )
name   = pd.get_dummies( carData[ 'carName' ], drop_first = True )
# Add the results to the original carData dataframe



carData = pd.concat([ carData, gas ], axis = 1)

carData = pd.concat([ carData, body ], axis = 1)

carData = pd.concat([ carData, engine ], axis = 1)

carData = pd.concat([ carData, wheels ], axis = 1)

carData = pd.concat([ carData, asp ], axis = 1)

carData = pd.concat([ carData, location ], axis = 1)

carData = pd.concat([ carData, system ], axis = 1)

carData = pd.concat([ carData, name ], axis = 1)
carData.head( )
carData.drop( [ 'fueltype', 'carbody', 'enginetype', 'drivewheel', 'aspiration', 'enginelocation', 'fuelsystem', 'carName' ], axis = 1, inplace = True)
carData.head( )
carData.shape
carData.drop( ['car_ID'], axis = 1, inplace = True )
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

dfTrain, dfTest = train_test_split( carData, train_size = 0.7, test_size = 0.3, random_state = 100 )
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables, 

numericCols = ['enginesize','compressionratio','citympg','highwaympg', 'wheelbase','carwidth', 

               'curbweight', 'horsepower', 'stroke','peakrpm', 'carlength', 'price', 'carheight',

               'boreratio', 'symboling', 'doornumber', 'cylindernumber' ]

dfTrain[ numericCols ] = scaler.fit_transform( dfTrain[ numericCols ] )
dfTrain.head()
dfTrain.describe( )
y_train = dfTrain.pop('price')

X_train = dfTrain
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
#Creating the Linear Regression Model 

lm = LinearRegression()

lm.fit( X_train, y_train )

# Running RFE with the output number of the variable equal to 15

rfe = RFE( lm, 15 )             # running RFE

rfe = rfe.fit( X_train, y_train )
#RFE Ranking of the features

list( zip( X_train.columns,rfe.support_,rfe.ranking_ ) )
#Columns that are to be used for building the model.

col = X_train.columns[ rfe.support_ ]

col
#Columns that are not part of the top '15' features

X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[ col ]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant( X_train_rfe )
lm = sm.OLS( y_train,X_train_rfe ).fit( )   # Running the linear model
#summary of our linear model

print( lm.summary( ) )
X_train_new = X_train_rfe.drop( [ "compressionratio" ], axis = 1 )
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant( X_train_new )

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
X_train_new = X_train_new.drop( [ "carlength" ], axis = 1 )

# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant( X_train_new )

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print( lm.summary( ) )
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_lm

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop( [ "enginesize" ], axis = 1 )

# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant( X_train_new )

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print( lm.summary( ) )
X_train_new = X_train_new.drop( [ "boreratio" ], axis = 1 )

# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant( X_train_new )

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print( lm.summary( ) )
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_lm

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop( [ "peugeot" ], axis = 1 )

# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant( X_train_new )

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print( lm.summary( ) )
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_lm

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop( [ "stroke", 'idi' ], axis = 1 )

# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant( X_train_new )

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print( lm.summary( ) )  
X_train_new = X_train_new.drop( [ "peakrpm" ], axis = 1 )

# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant( X_train_new )

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print( lm.summary( ) )
X_train_new = X_train_new.drop( [ "porsche" ], axis = 1 )

# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant( X_train_new )

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print( lm.summary( ) )
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_lm

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop( [ 'const' ], axis=1 )
#building the predicted price

y_train_price = lm.predict( X_train_lm )
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
#Scaling the columns, similar to the columns scaled in the training data set

dfTest[ numericCols ] = scaler.transform( dfTest[ numericCols ] )
y_test = dfTest.pop('price')

X_test = dfTest
# Creating X_test_new dataframe by dropping variables from X_test which were not used to build the model.

X_test_new = X_test[ X_train_new.columns ]



# Adding a constant variable 

X_test_new = sm.add_constant( X_test_new )
lm = sm.OLS( y_test, X_test_new ).fit( )   # Running the linear model

print( lm.summary( ) ) 
# Making predictions on the price value using the test data set.

y_pred = lm.predict( X_test_new )
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure( )

plt.scatter( y_test, y_pred )

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot(( y_test - y_pred ), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
r_squared = r2_score( y_test, y_pred )

r_squared