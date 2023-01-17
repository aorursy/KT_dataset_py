# import all libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import re



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import scale

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline



import warnings # supress warnings

warnings.filterwarnings('ignore')
# import Housing.csv

housing = pd.read_csv('../input/cross-val/Housing.csv')

housing.head()
# number of observations 

len(housing.index)
# filter only area and price

df = housing.loc[:, ['area', 'price']]

df.head()
# recaling the variables (both)

df_columns = df.columns

scaler = MinMaxScaler()

df = scaler.fit_transform(df)



# rename columns (since now its an np array)

df = pd.DataFrame(df)

df.columns = df_columns



df.head()
# visualise area-price relationship

sns.regplot(x="area", y="price", data=df, fit_reg=False)
# split into train and test

df_train, df_test = train_test_split(df, 

                                     train_size = 0.7, 

                                     test_size = 0.3, 

                                     random_state = 10)

print(len(df_train))

print(len(df_test))
# split into X and y for both train and test sets

# reshaping is required since sklearn requires the data to be in shape

# (n, 1), not as a series of shape (n, )

X_train = df_train['area']

X_train = X_train.values.reshape(-1, 1)

y_train = df_train['price']



X_test = df_test['area']

X_test = X_test.values.reshape(-1, 1)

y_test = df_test['price']
len(X_train)
# fit multiple polynomial features

degrees = [1, 2, 3, 6, 10, 20]



# initialise y_train_pred and y_test_pred matrices to store the train and test predictions

# each row is a data point, each column a prediction using a polynomial of some degree

y_train_pred = np.zeros((len(X_train), len(degrees)))

y_test_pred = np.zeros((len(X_test), len(degrees)))



for i, degree in enumerate(degrees):

    

    # make pipeline: create features, then feed them to linear_reg model

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    model.fit(X_train, y_train)

    

    # predict on test and train data

    # store the predictions of each degree in the corresponding column

    y_train_pred[:, i] = model.predict(X_train)

    y_test_pred[:, i] = model.predict(X_test)

    
# visualise train and test predictions

# note that the y axis is on a log scale



plt.figure(figsize=(16, 8))



# train data

plt.subplot(121)

plt.scatter(X_train, y_train)

plt.yscale('log')

plt.title("Train data")

for i, degree in enumerate(degrees):    

    plt.scatter(X_train, y_train_pred[:, i], s=15, label=str(degree))

    plt.legend(loc='upper left')

    

# test data

plt.subplot(122)

plt.scatter(X_test, y_test)

plt.yscale('log')

plt.title("Test data")

for i, degree in enumerate(degrees):    

    plt.scatter(X_test, y_test_pred[:, i], label=str(degree))

    plt.legend(loc='upper left')
# compare r2 for train and test sets (for all polynomial fits)

print("R-squared values: \n")



for i, degree in enumerate(degrees):

    train_r2 = round(sklearn.metrics.r2_score(y_train, y_train_pred[:, i]), 2)

    test_r2 = round(sklearn.metrics.r2_score(y_test, y_test_pred[:, i]), 2)

    print("Polynomial degree {0}: train score={1}, test score={2}".format(degree, 

                                                                         train_r2, 

                                                                         test_r2))
# data preparation



# list of all the "yes-no" binary categorical variables

# we'll map yes to 1 and no to 0

binary_vars_list =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']



# defining the map function

def binary_map(x):

    return x.map({'yes': 1, "no": 0})



# applying the function to the housing variables list

housing[binary_vars_list] = housing[binary_vars_list].apply(binary_map)

housing.head()
# 'dummy' variables

# get dummy variables for 'furnishingstatus' 

# also, drop the first column of the resulting df (since n-1 dummy vars suffice)

status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)

status.head()
# concat the dummy variable df with the main df

housing = pd.concat([housing, status], axis = 1)

housing.head()
# 'furnishingstatus' since we alreday have the dummy vars

housing.drop(['furnishingstatus'], axis = 1, inplace = True)

housing.head()
# train-test 70-30 split

df_train, df_test = train_test_split(housing, 

                                     train_size = 0.7, 

                                     test_size = 0.3, 

                                     random_state = 100)



# rescale the features

scaler = MinMaxScaler()



# apply scaler() to all the numeric columns 

numeric_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_train[numeric_vars] = scaler.fit_transform(df_train[numeric_vars])

df_train.head()
# apply rescaling to the test set also

df_test[numeric_vars] = scaler.fit_transform(df_test[numeric_vars])

df_test.head()
# divide into X_train, y_train, X_test, y_test

y_train = df_train.pop('price')

X_train = df_train



y_test = df_test.pop('price')

X_test = df_test
# num of max features

len(X_train.columns)
# first model with an arbitrary choice of n_features

# running RFE with number of features=10



lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, n_features_to_select=10)             

rfe = rfe.fit(X_train, y_train)
# tuples of (feature name, whether selected, ranking)

# note that the 'rank' is > 1 for non-selected features

list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# predict prices of X_test

y_pred = rfe.predict(X_test)



# evaluate the model on test set

r2 = sklearn.metrics.r2_score(y_test, y_pred)

print(r2)
# try with another value of RFE

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, n_features_to_select=6)             

rfe = rfe.fit(X_train, y_train)



# predict prices of X_test

y_pred = rfe.predict(X_test)

r2 = sklearn.metrics.r2_score(y_test, y_pred)

print(r2)
# k-fold CV (using all the 13 variables)

lm = LinearRegression()

scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=5)

scores      
# the other way of doing the same thing (more explicit)



# create a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=folds)

scores   
# can tune other metrics, such as MSE

scores = cross_val_score(lm, X_train, y_train, scoring='neg_mean_squared_error', cv=5)

scores
# number of features in X_train

len(X_train.columns)
# step-1: create a cross-validation scheme

folds = KFold(n_splits = 5, shuffle = True, random_state = 100)



# step-2: specify range of hyperparameters to tune

hyper_params = [{'n_features_to_select': list(range(1, 14))}]





# step-3: perform grid search

# 3.1 specify model

lm = LinearRegression()

lm.fit(X_train, y_train)

rfe = RFE(lm)             



# 3.2 call GridSearchCV()

model_cv = GridSearchCV(estimator = rfe, 

                        param_grid = hyper_params, 

                        scoring= 'r2', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



# fit the model

model_cv.fit(X_train, y_train)                  

# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# plotting cv results

plt.figure(figsize=(16,6))



plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])

plt.xlabel('number of features')

plt.ylabel('r-squared')

plt.title("Optimal Number of Features")

plt.legend(['test score', 'train score'], loc='upper left')
# final model

n_features_optimal = 10



lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, n_features_to_select=n_features_optimal)             

rfe = rfe.fit(X_train, y_train)



# predict prices of X_test

y_pred = lm.predict(X_test)

r2 = sklearn.metrics.r2_score(y_test, y_pred)

print(r2)
# reading the dataset

cars = pd.read_csv("../input/cross-val/CarPrice_Assignment.csv")
# All data preparation steps in this cell



# converting symboling to categorical

cars['symboling'] = cars['symboling'].astype('object')





# create new column: car_company

p = re.compile(r'\w+-?\w+')

cars['car_company'] = cars['CarName'].apply(lambda x: re.findall(p, x)[0])





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





# drop carname variable

cars = cars.drop('CarName', axis=1)





# split into X and y

X = cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',

       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',

       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',

       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',

       'horsepower', 'peakrpm', 'citympg', 'highwaympg',

       'car_company']]

y = cars['price']





# creating dummy variables for categorical variables

cars_categorical = X.select_dtypes(include=['object'])

cars_categorical.head()





# convert into dummies

cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)

cars_dummies.head()





# drop categorical variables 

X = X.drop(list(cars_categorical.columns), axis=1)





# concat dummy variables with X

X = pd.concat([X, cars_dummies], axis=1)





# rescale the features

cols = X.columns

X = pd.DataFrame(scale(X))

X.columns = cols





# split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    train_size=0.7,

                                                    test_size = 0.3, random_state=40)
# number of features

len(X_train.columns)
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 100)



# specify range of hyperparameters

hyper_params = [{'n_features_to_select': list(range(2, 40))}]



# specify model

lm = LinearRegression()

lm.fit(X_train, y_train)

rfe = RFE(lm)             



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = rfe, 

                        param_grid = hyper_params, 

                        scoring= 'r2', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



# fit the model

model_cv.fit(X_train, y_train)                  
# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# plotting cv results

plt.figure(figsize=(16,6))



plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])

plt.xlabel('number of features')

plt.ylabel('r-squared')

plt.title("Optimal Number of Features")

plt.legend(['test score', 'train score'], loc='upper left')