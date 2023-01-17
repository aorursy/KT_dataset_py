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
import seaborn as sns
import matplotlib.pyplot as plt
cars_data = pd.read_csv(r'/kaggle/input/used-cars-database/autos.csv', encoding='ISO-8859-1')
cars_data.head(5)
cars_data.shape
## Create copy of dataframe for data manipulation
cars_copy=cars_data.copy()
cars_copy.info()
cars_copy.describe()
## To display maximum set of columns
pd.set_option('display.max_columns', 500)
cars_copy.describe()
## Dropping unnecessary columns from dataframe
col=['name','dateCrawled','dateCreated','postalCode','lastSeen','nrOfPictures']
cars_copy = cars_copy.drop(columns=col, axis=1)
cars_copy.shape
## Remove duplicate records
cars_copy.drop_duplicates(keep='first', inplace=True)
cars_copy.shape
## Find total null values in each column
cars_copy.isnull().sum()
## Variable year of registration
yearwise_count=cars_copy['yearOfRegistration'].value_counts().sort_index()
sum(cars_copy['yearOfRegistration'] >2020)
sum(cars_copy['yearOfRegistration'] >1950)
sns.regplot(x='yearOfRegistration', y='price', scatter=True, fit_reg=False, data=cars_copy)
## Working range 1950 and 2020
## No. of cars having year of registration is greater than 2020 thats why above graph is unclear due to higher value
yearwise_count=cars_copy['yearOfRegistration'].value_counts().sort_index()
sum(cars_copy['yearOfRegistration'] >2020)
## No. of cars having year of registration is lesser than 1950
yearwise_count=cars_copy['yearOfRegistration'].value_counts().sort_index()
sum(cars_copy['yearOfRegistration'] <1950)
## Now work on variable price for cleaning
price_count=cars_copy['price'].value_counts().sort_index()
price_count
## Now working range is 100 to 150000
sns.distplot(cars_copy['price'])
cars_copy['price'].describe()
## check how many cars price is out of our range i.e 100 to 150000
sns.boxplot(y=cars_copy['price'])
sum(cars_copy['price'] >150000)
sum(cars_copy['price'] <100)
## The box plot is unclear coz of extreme values in data which is highly extreme
## Next variable is powerPS
power_count =cars_copy['powerPS'].value_counts().sort_index()
power_count
sns.distplot(cars_copy['powerPS'])
## Above distplot is unclear and irregular due to unconsistent value in column
cars_copy['powerPS'].describe()
## Boxplot is unclear and irregular due to unconsistent value in column
sns.boxplot(y=cars_copy['powerPS'])
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars_copy)
## Now set range of powerPS in column i.e 10 to 500 and check how many values are out of range
sum(cars_copy['powerPS'] >500)
sum(cars_copy['powerPS'] <100)
## Set range of coluun values of dataframe for further processing 
cars_copy = cars_copy[
    (cars_copy.yearOfRegistration <=2020)
   & (cars_copy.yearOfRegistration >=1950)
   & (cars_copy.price >=100)
    & (cars_copy.price <= 150000)
    & (cars_copy.powerPS >= 10)
    & (cars_copy.powerPS <=500)
     ]
cars_copy.shape ##Sure we loose some data which is out of range
## Further simplication to reduce number of variables
## Combining year of registration and month of registration 

cars_copy['monthOfRegistration']/=12
## Creating new variable Age by adding yearOfRegistration and monthOfRegistration
cars_copy['Age'] = (2020-cars_copy['yearOfRegistration']) + cars_copy['monthOfRegistration']
cars_copy['Age'] = round(cars_copy['Age'],2)
cars_copy['Age'].describe()
## Now, drop column yearofregistration and monthofregistration from df
cars_copy.drop(columns=['yearOfRegistration', 'monthOfRegistration'], axis =1, inplace=True)
cars_copy.shape
sns.distplot(cars_copy['Age'])
plt.title("Age Frequency of cars")
plt.show()
## boxpllot of Age of an car it shows min, max and outliners or extreme age value of an car
sns.boxplot(y =cars_copy['Age'])
plt.show()
sns.distplot(cars_copy['price'])
plt.show()
sns.boxplot(y= cars_copy['price'])
plt.show
## PowerPS plot representation
sns.distplot(cars_copy['powerPS'])
plt.show()
sns.boxplot(y =cars_copy['powerPS'])
## Visualizing parameters after narrowing working range 
sns.regplot(x='Age', y='price', scatter =True, data=cars_copy)
sns.regplot(x='powerPS', y='price', scatter=True, data=cars_copy, fit_reg=True)
## Check individual frequency count of an category under categorical variables 
## Variable seller'
cars_copy['seller'].value_counts()
pd.crosstab(cars_copy['seller'], columns ='count', normalize=True) ## FInd marginal probability
sns.countplot(x='seller', data=cars_copy)
## Variable offerType
cars_copy['offerType'].value_counts()
sns.countplot(x='offerType', data=cars_copy)
## VAriable abtest
cars_copy['abtest'].value_counts()
sns.countplot(x='abtest', data=cars_copy)
pd.crosstab(cars_copy['abtest'], columns='count', normalize=True)
## Equally distributed
sns.boxplot(x= 'abtest', y='price', data=cars_copy)
## Variable Vehicletype
cars_copy['vehicleType'].value_counts()
pd.crosstab(cars_copy['vehicleType'], columns ='count', normalize=True)
## Which type of car are in higher number
sns.countplot(x='vehicleType', data=cars_copy)
plt.xticks(rotation='vertical')
plt.show()
## Show box plot of an car of different type
sns.boxplot(x= 'vehicleType',y='price', data=cars_copy)
## Vehicle gearbox
cars_copy['gearbox'].value_counts()
pd.crosstab(cars_copy['gearbox'], columns='count', normalize=True)
sns.countplot(x='gearbox', data=cars_copy)
## Find impact of different gearbox in price of a car
sns.boxplot(x='gearbox', y='price', data=cars_copy)
## Variable model
cars_copy['model'].value_counts() >10000
## Totally irrelevant to analyse because of large no. of category of car models
sns.boxplot(x='model', y='price', data=cars_copy)
## Variable kilometer
cars_copy['kilometer'].value_counts()
pd.crosstab(cars_copy['kilometer'], columns='count', normalize=True)
sns.boxplot(x='kilometer', y='price', data=cars_copy)
plt.xticks(rotation='vertical')
plt.show()
## Variable Fueltype
cars_copy['fuelType'].value_counts()
pd.crosstab(cars_copy['fuelType'], columns='count', normalize=True)
sns.boxplot(x='fuelType', y='price', data=cars_copy)
## VAriable Brand
cars_copy.brand.value_counts()
sns.boxplot(x='brand', y= 'price', data=cars_copy)
plt.xticks(rotation='vertical')
plt.show()
## Variable notRepairDamage
## yes - car is damaged but not rectified
## no - car is damaged but has been rectified
cars_copy.notRepairedDamage.value_counts()
pd.crosstab(cars_copy['notRepairedDamage'], columns='count', normalize=True)
sns.boxplot(x= 'notRepairedDamage', y='price', data=cars_copy)
## Recovering insignificant variables
col_insig =['abtest','seller','offerType']
cars_copy =cars_copy.drop(columns=col_insig, axis=1)
cars_copy2 = cars_copy.copy()
## Correlation between numeric data type variable only
cars_select1 =cars_copy.select_dtypes(exclude='object')
correlation =cars_select1.corr()
round(correlation,3)
## PLot heatmap of correlation matrix
sns.heatmap(correlation,  cmap="YlGnBu")
## .loc function based on price and return only 1st column and taking absolute value and sorting in ascending order
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
cars_omit = cars_copy.dropna(axis=0)
## After drop missing rows we check shape of data
cars_omit.shape
## Converting categorical variables to dummy variables
cars_omit=pd.get_dummies(cars_omit, drop_first=True)
## After adding dummy values the columns will be increase coz each category is now shown in either 0 or 1 and there are large number of categories in categorical varaible
cars_omit.shape
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
## Separating input and output features
x1 = cars_omit.drop(['price'], axis='columns', inplace=False)
y1 = cars_omit['price']
## Plotting the variable price and log of price
prices =pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
prices.hist()
plt.show()
## Transform price as a logarithmic value
y1 =np.log(y1)
## splitting data into test and train to fit model & predict
## Train set contains 70% data because test_size =0.3 and random state is a predefined algorithm its called pseudo random  number generator 
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
## Finding the mean for test data value
base_pred =np.mean(y_test)
print(base_pred)
## Representing some value till length of test data
base_pred = np.repeat(base_pred, len(y_test))
## FInding the RMSE(Root Mean Squared Error)
## RMSE computes the difference between the test value and the predicted value and squared them and divides them by number of samples.

base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print(base_root_mean_square_error)
## Setting intercept as true
lgr = LinearRegression(fit_intercept =True)
## MODEL
model_lin1 = lgr.fit(x_train, y_train)
## Predicting model on test set
cars_predictions_lin1 = lgr.predict(x_test)
## Computing MSE and RMSE
lin_mse1 = mean_squared_error(y_test, cars_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)
## R squared value
r2_lin_test1 = model_lin1.score(x_test, y_test)
r2_lin_train1 = model_lin1.score(x_train, y_train)
print(r2_lin_test1, r2_lin_train1)
## Regression diagnostics :- Resident plot analysis
## It is differnce test data and your prediction. It is just difference between actual & predicted value.
residuals1 = y_test - cars_predictions_lin1
sns.regplot(x = cars_predictions_lin1, y=residuals1, scatter=True, fit_reg=False, data=cars_copy)
residuals1.describe()
## MODEL PARAMETERS
rf = RandomForestRegressor(n_estimators = 100, max_features='auto', max_depth=100, min_samples_split=10, min_samples_leaf=4, random_state=1)
## MODEL
model_rf1 =rf.fit(x_train, y_train)
## Predicting model on test set
cars_predictions_rf1 = rf.predict(x_test)
## Computing MSE and RSME
rf_mse1 = mean_squared_error(y_test, cars_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)
## R Squared value
r2_rf_test1 = model_rf1.score(x_test, y_test)
r2_rf_train1 = model_rf1.score(x_train, y_train)
print(r2_rf_test1, r2_rf_train1)
## Fillna will fill missing value with median in float data type variable and otherwise it will fill the cell with most frequent value

cars_inputed = cars_copy.apply(lambda x:x.fillna(x.median()) if x.dtype=='float' else x.fillna(x.value_counts().index[0]))
cars_inputed.isnull().sum()
## Converting categorical variables to dummy variables

cars_inputed = pd.get_dummies(cars_inputed, drop_first = True)
## MODEL BUILDING

## Separating input and output feature
x2 = cars_inputed.drop(['price'], axis='columns', inplace=False)
y2 = cars_inputed['price']
## Ploting the variable price
prices2 = pd.DataFrame({"1. Before":y2, "2. After": np.log(y2)})
## Transforming price as a logarithmic value
y2 = np.log(y2)
## splitting data into test and train to fit model & predict
## Train set contains 70% data because test_size =0.3 and random state is a predefined algorithm its called pseudo random  number generator 

x_train1, x_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size=0.3, random_state = 3)
print(x_train1.shape, x_test1.shape, y_train1.shape, y_test1.shape)
## Find the mean for test data
base_pred2 = np.mean(y_test1)
print(base_pred2)
## Representing some value till length of test data
base_pred2 = np.repeat(base_pred2, len(y_test1))
## FInding the RMSE(Root Mean Squared Error)
## RMSE computes the difference between the test value and the predicted value and squared them and divides them by number of samples.

base_root_mean_square_error_inputed = np.sqrt(mean_squared_error(y_test1, base_pred2))
print(base_root_mean_square_error_inputed)
## Setting intercept as true
lgr2 = LinearRegression(fit_intercept =True)
## MODEL
model_lin2 = lgr2.fit(x_train1, y_train1)
## Predicting model on test set
cars_predictions_lin2 = lgr2.predict(x_test1)
## Computing MSE and RMSE
lin_mse2 = mean_squared_error(y_test1, cars_predictions_lin2)
lin_rmse2 = np.sqrt(lin_mse2)
print(lin_rmse2)
## R squared value
r2_lin_test2 = model_lin2.score(x_test1, y_test1)
r2_lin_train2 = model_lin2.score(x_train1, y_train1)
print(r2_lin_test2, r2_lin_train2)
## MODEL PARAMETERS
rf2 = RandomForestRegressor(n_estimators = 100, max_features='auto', max_depth=100, min_samples_split=10, min_samples_leaf=4, random_state=1)
## MODEL
model_rf2 =rf2.fit(x_train1, y_train1)
## Predicting model on test set
cars_predictions_rf2 = rf2.predict(x_test1)
## Computing MSE and RSME
rf_mse2 = mean_squared_error(y_test1, cars_predictions_rf2)
rf_rmse2 = np.sqrt(rf_mse2)
print(rf_rmse2)
## R Squared value
r2_rf_test2 = model_rf2.score(x_test1, y_test1)
r2_rf_train2 = model_rf2.score(x_train1, y_train1)
print(r2_rf_test2, r2_rf_train2)