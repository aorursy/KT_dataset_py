import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression



# reading the dataset

cars = pd.read_csv("../input/car-data/CarPrice_Assignment.csv")
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

sns.distplot(cars['wheelbase'])

plt.show()
# curbweight: weight of car without occupants or baggage

sns.distplot(cars['curbweight'])

plt.show()
# stroke: volume of the engine (the distance traveled by the 

# piston in each cycle)

sns.distplot(cars['stroke'])

plt.show()
# compression ration: ration of volume of compression chamber 

# at largest capacity to least capacity

sns.distplot(cars['compressionratio'])

plt.show()
# target variable: price of car

sns.distplot(cars['price'])

plt.show()
# all numeric (float and int) variables in the dataset

cars_numeric = cars.select_dtypes(include=['float64', 'int'])

cars_numeric.head()
# dropping symboling and car_ID 

cars_numeric = cars_numeric.drop(['symboling', 'car_ID'], axis=1)

cars_numeric.head()
# paiwise scatter plot



plt.figure(figsize=(20, 10))

sns.pairplot(cars_numeric)

plt.show()
# correlation matrix

cor = cars_numeric.corr()

cor
# plotting correlations on a heatmap



# figure size

plt.figure(figsize=(16,8))



# heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()

# variable formats

cars.info()
# converting symboling to categorical

cars['symboling'] = cars['symboling'].astype('object')

cars.info()
# CarName: first few entries

cars['CarName'][:30]
# Extracting carname



# Method 1: str.split() by space

carnames = cars['CarName'].apply(lambda x: x.split(" ")[0])

carnames[:30]
# Method 2: Use regular expressions

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
# scaling the features

from sklearn.preprocessing import scale



# storing column names in cols, since column names are (annoyingly) lost after 

# scaling (the df is converted to a numpy array)

cols = X.columns

X = pd.DataFrame(scale(X))

X.columns = cols

X.columns
# split into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    train_size=0.7,

                                                    test_size = 0.3, random_state=100)
# Building the first model with all the features



# instantiate

lm = LinearRegression()



# fit

lm.fit(X_train, y_train)
# print coefficients and intercept

print(lm.coef_)

print(lm.intercept_)
# predict 

y_pred = lm.predict(X_test)



# metrics

from sklearn.metrics import r2_score



print(r2_score(y_true=y_test, y_pred=y_pred))
# RFE with 15 features

from sklearn.feature_selection import RFE



# RFE with 15 features

lm = LinearRegression()

rfe_15 = RFE(lm, 15)



# fit with 15 features

rfe_15.fit(X_train, y_train)



# Printing the boolean results

print(rfe_15.support_)           

print(rfe_15.ranking_)  
# making predictions using rfe model

y_pred = rfe_15.predict(X_test)



# r-squared

print(r2_score(y_test, y_pred))
# RFE with 6 features

from sklearn.feature_selection import RFE



# RFE with 6 features

lm = LinearRegression()

rfe_6 = RFE(lm, 6)



# fit with 6 features

rfe_6.fit(X_train, y_train)



# predict

y_pred = rfe_6.predict(X_test)



# r-squared

print(r2_score(y_test, y_pred))
# import statsmodels

import statsmodels.api as sm  



# subset the features selected by rfe_15

col_15 = X_train.columns[rfe_15.support_]



# subsetting training data for 15 selected columns

X_train_rfe_15 = X_train[col_15]



# add a constant to the model

X_train_rfe_15 = sm.add_constant(X_train_rfe_15)

X_train_rfe_15.head()
# fitting the model with 15 variables

lm_15 = sm.OLS(y_train, X_train_rfe_15).fit()   

print(lm_15.summary())
# making predictions using rfe_15 sm model

X_test_rfe_15 = X_test[col_15]





# # Adding a constant variable 

X_test_rfe_15 = sm.add_constant(X_test_rfe_15, has_constant='add')

X_test_rfe_15.info()





# # Making predictions

y_pred = lm_15.predict(X_test_rfe_15)

# r-squared

r2_score(y_test, y_pred)
# subset the features selected by rfe_6

col_6 = X_train.columns[rfe_6.support_]



# subsetting training data for 6 selected columns

X_train_rfe_6 = X_train[col_6]



# add a constant to the model

X_train_rfe_6 = sm.add_constant(X_train_rfe_6)





# fitting the model with 6 variables

lm_6 = sm.OLS(y_train, X_train_rfe_6).fit()   

print(lm_6.summary())





# making predictions using rfe_6 sm model

X_test_rfe_6 = X_test[col_6]





# Adding a constant  

X_test_rfe_6 = sm.add_constant(X_test_rfe_6, has_constant='add')

X_test_rfe_6.info()





# # Making predictions

y_pred = lm_6.predict(X_test_rfe_6)
# r2_score for 6 variables

r2_score(y_test, y_pred)
n_features_list = list(range(4, 20))

adjusted_r2 = []

r2 = []

test_r2 = []



for n_features in range(4, 20):



    # RFE with n features

    lm = LinearRegression()



    # specify number of features

    rfe_n = RFE(lm, n_features)



    # fit with n features

    rfe_n.fit(X_train, y_train)



    # subset the features selected by rfe_6

    col_n = X_train.columns[rfe_n.support_]



    # subsetting training data for 6 selected columns

    X_train_rfe_n = X_train[col_n]



    # add a constant to the model

    X_train_rfe_n = sm.add_constant(X_train_rfe_n)





    # fitting the model with 6 variables

    lm_n = sm.OLS(y_train, X_train_rfe_n).fit()

    adjusted_r2.append(lm_n.rsquared_adj)

    r2.append(lm_n.rsquared)

    

    

    # making predictions using rfe_15 sm model

    X_test_rfe_n = X_test[col_n]





    # # Adding a constant variable 

    X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')







    # # Making predictions

    y_pred = lm_n.predict(X_test_rfe_n)

    

    test_r2.append(r2_score(y_test, y_pred))

# plotting adjusted_r2 against n_features

plt.figure(figsize=(10, 8))

plt.plot(n_features_list, adjusted_r2, label="adjusted_r2")

plt.plot(n_features_list, r2, label="train_r2")

plt.plot(n_features_list, test_r2, label="test_r2")

plt.legend(loc='upper left')

plt.show()
# RFE with n features

lm = LinearRegression()



n_features = 6



# specify number of features

rfe_n = RFE(lm, n_features)



# fit with n features

rfe_n.fit(X_train, y_train)



# subset the features selected by rfe_6

col_n = X_train.columns[rfe_n.support_]



# subsetting training data for 6 selected columns

X_train_rfe_n = X_train[col_n]



# add a constant to the model

X_train_rfe_n = sm.add_constant(X_train_rfe_n)





# fitting the model with 6 variables

lm_n = sm.OLS(y_train, X_train_rfe_n).fit()

adjusted_r2.append(lm_n.rsquared_adj)

r2.append(lm_n.rsquared)





# making predictions using rfe_15 sm model

X_test_rfe_n = X_test[col_n]





# # Adding a constant variable 

X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')







# # Making predictions

y_pred = lm_n.predict(X_test_rfe_n)



test_r2.append(r2_score(y_test, y_pred))
# summary

lm_n.summary()
# results 

r2_score(y_test, y_pred)
# Error terms

c = [i for i in range(len(y_pred))]

fig = plt.figure()

plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('ytest-ypred', fontsize=16)                # Y-label

plt.show()
# Plotting the error terms to understand the distribution.

fig = plt.figure()

sns.distplot((y_test-y_pred),bins=50)

fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 

plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label

plt.ylabel('Index', fontsize=16)                          # Y-label

plt.show()
# mean

np.mean(y_test-y_pred)

sns.distplot(cars['price'],bins=50)

plt.show()
# multicollinearity

predictors = ['carwidth', 'curbweight', 'enginesize', 

             'enginelocation_rear', 'car_company_bmw', 'car_company_porsche']



cors = X.loc[:, list(predictors)].corr()

sns.heatmap(cors, annot=True)

plt.show()