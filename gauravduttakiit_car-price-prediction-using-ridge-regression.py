import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

import os



# hide warnings

import warnings

warnings.filterwarnings('ignore')

# reading the dataset

cars = pd.read_csv(r'/kaggle/input/car-price/CarPrice_Assignment.csv')

# summary of the dataset: 205 rows, 26 columns, no null values

print(cars.info())
# head

cars.head()
# symboling: -2 (least risky) to +3 most risky

# Most cars are 0,1,2

cars['symboling'].astype('category').value_counts()



# aspiration: An (internal combustion) engine property showing 

# whether the oxygen intake is through standard (atmospheric pressure)

# or through turbocharging (pressurised oxygen intake)



cars['aspiration'].astype('category').value_counts()
# drivewheel: frontwheel, rarewheel or four-wheel drive 

cars['drivewheel'].astype('category').value_counts()
# wheelbase: distance between centre of front and rarewheels

plt.figure(figsize=(20,5))

sns.distplot(cars['wheelbase'])

plt.show()
# curbweight: weight of car without occupants or baggage

plt.figure(figsize=(20,5))

sns.distplot(cars['curbweight'])

plt.show()
# stroke: volume of the engine (the distance traveled by the piston in each cycle)

plt.figure(figsize=(20,5))

sns.distplot(cars['stroke'])

plt.show()
# compression ration: ratio of volume of compression chamber at largest capacity to least capacity

plt.figure(figsize=(20,5))

sns.distplot(cars['compressionratio'])

plt.show()
# target variable: price of car

plt.figure(figsize=(20,5))

sns.distplot(cars['price'])

plt.show()
# all numeric (float and int) variables in the dataset

cars_numeric = cars.select_dtypes(include=['float', 'int'])

cars_numeric.head()
# dropping symboling and car_ID 

cars_numeric = cars_numeric.drop(['symboling','car_ID'], axis=1)

cars_numeric.head()
#paiwise scatter plot

sns.pairplot(cars_numeric)

plt.show()
# correlation matrix

cor = cars_numeric.corr()

cor
# plotting correlations on a heatmap



# figure size

plt.figure(figsize=(16,8))



# heatmap

sns.heatmap(cor, cmap="rainbow", annot=True)

plt.show()

# variable formats

cars.info()
# converting symboling to categorical

cars['symboling'] = cars['symboling'].astype('object')

cars.info()
# CarName: first few entries

cars['CarName'][:30]
# Extracting carname



#str.split() by space

carnames = cars['CarName'].apply(lambda x: x.split(" ")[0])

carnames[:30]
import re



# regex: any alphanumeric sequence before a space, may contain a hyphen

p = re.compile(r'\w+-?\w+')

carnames = cars['CarName'].apply(lambda x: re.findall(p, x)[0])

print(carnames)
# New column car_company

cars['car_company'] = cars['CarName'].apply(lambda x: re.findall(p, x)[0])
# look at all values 

cars['car_company'].astype('category').value_counts()
# replacing misspelled car_company names



# volkswagen

cars.loc[(cars['car_company'] == "vw") | 

         (cars['car_company'] == "vokswagen")

         , 'car_company'] = 'volkswagen'



# porsche

cars.loc[cars['car_company'] == "porcshce", 'car_company'] = 'porsche'



# toyota

cars.loc[cars['car_company'] == "toyouta", 'car_company'] = 'toyota'



# nissan

cars.loc[cars['car_company'] == "Nissan", 'car_company'] = 'nissan'



# mazda

cars.loc[cars['car_company'] == "maxda", 'car_company'] = 'mazda'
cars['car_company'].astype('category').value_counts()
# drop carname variable

cars = cars.drop('CarName', axis=1)
cars.info()
# outliers

cars.describe()
cars.info()
# split into X and y

X = cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',

       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',

       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',

       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',

       'horsepower', 'peakrpm', 'citympg', 'highwaympg',

       'car_company']]



y = cars['price']

# creating dummy variables for categorical variables



# subset all categorical variables

cars_categorical = X.select_dtypes(include=['object'])

cars_categorical.head()

# convert into dummies

cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)

cars_dummies.head()
# drop categorical variables 

X = X.drop(list(cars_categorical.columns), axis=1)
# concat dummy variables with X

X = pd.concat([X, cars_dummies], axis=1)
# split into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    train_size=0.7,

                                                    test_size = 0.3, random_state=100)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',

       'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',

       'peakrpm', 'citympg', 'highwaympg']]=scaler.fit_transform(X_train[['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',

       'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',

       'peakrpm', 'citympg', 'highwaympg']])

X_train.head()
X_test[['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',

       'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',

       'peakrpm', 'citympg', 'highwaympg']]=scaler.transform(X_test[['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',

       'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',

       'peakrpm', 'citympg', 'highwaympg']])

X_test.head()
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}





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
cv_results = pd.DataFrame(model_cv.cv_results_)



cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int')



# plotting

plt.figure(figsize=(20,10))

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.grid()

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
cv_results = cv_results[cv_results['param_alpha']<=200]
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int')



# plotting

plt.figure(figsize=(20,10))

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.grid()

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
alpha = 15

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
imp_ridge = pd.DataFrame({

    "Varname": X_train.columns,

    "Coefficient": ridge.coef_})

imp_ridge.sort_values(by="Coefficient", ascending=False)
imp_ridge=imp_ridge.drop([imp_ridge.index[56], imp_ridge.index[43],imp_ridge.index[46]])

imp_ridge.sort_values(by="Coefficient", ascending=False)
y_pred = ridge.predict(X_test)
fig = plt.figure(figsize=(20,10))

plt.scatter(y_test, y_pred, alpha=.5)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16) 

plt.show()
df= pd.DataFrame({'Actual':y_test,'Predictions':y_pred})

df['Predictions']= round(df['Predictions'],2)

df.head()
from sklearn import metrics 
metrics.explained_variance_score(y_test,y_pred)
metrics.mean_absolute_error(y_test,y_pred)
metrics.max_error(y_test,y_pred)
metrics.mean_squared_error(y_test,y_pred)
metrics.mean_squared_log_error(y_test,y_pred)
metrics.median_absolute_error(y_test,y_pred)
metrics.r2_score(y_test,y_pred)
metrics.mean_poisson_deviance(y_test,y_pred)
metrics.mean_gamma_deviance(y_test,y_pred)
metrics.mean_tweedie_deviance(y_test,y_pred)