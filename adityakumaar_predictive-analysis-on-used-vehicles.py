# Importing required libraries

import pandas as pd



# Loading the dataset

#directory = "../input/used-vehicles-prices.csv"

df_vehicle = pd.read_csv("../input/used-vehicles-prices/vehicle_dataset.csv")
# Displaying the shape and size of the dataset

print("Shape of the dataset: ", df_vehicle.shape)

print("Size of the dataset: ", df_vehicle.size)
# Displaying the column names for the dataset

df_vehicle.columns
# Displaying the top 5 rows from the dataset

df_vehicle.head()
# Displaying unique values in each columns

print("Unique values for Seller Type:     ", df_vehicle['Seller_Type'].unique())

print("Unique values for Fuel Type:       ", df_vehicle['Fuel_Type'].unique())

print("Unique values for Transmission:    ", df_vehicle['Transmission'].unique())

print("Unique values for Previous owners: ", df_vehicle['Owner'].unique())
# Checking for missing values in each column

df_vehicle.isnull().sum()
# Statistical description of the dataset

df_vehicle.describe(include = "all")
# Selecting required features for the final dataset

final_dataset = df_vehicle[['Year',

                            'Selling_Price',

                            'Present_Price',

                            'Kms_Driven',

                            'Fuel_Type',

                            'Seller_Type',

                            'Transmission',

                            'Owner'

                           ]]



# Displaying top 5 columns of the final_dataset

final_dataset.head()
# Sorting the dataset having current year as 2020

final_dataset['Current Year'] = 2020



# Displaying the dataset after sorting

final_dataset.head()
# Finding the number of years after the release of the vehicle

final_dataset['no_year'] = final_dataset['Current Year'] - final_dataset['Year']



# Displaying the top 5 values from the dataset

final_dataset.head()
# Dropping the release year column as we dont need it now

final_dataset.drop(['Year'], axis = 1, inplace = True)



# Displaying the top 5 values from the dataset

final_dataset.head()
# Setting dummy values for the features

final_dataset = pd.get_dummies(final_dataset, drop_first = True)



# Displaying the top 5 rows with dummy values

final_dataset.head()
# Dropping the Current Year column from the dataset

final_dataset = final_dataset.drop(['Current Year'], axis = 1)



# Displaying the top 5 rows of the modifies dataset

final_dataset.head()
# Creating a correlational table for the final variables

final_dataset.corr()
# Importing seaborn for plotting graphs

import seaborn as sns



# Plotting a pairplot for the final variables

sns.pairplot(final_dataset)
# Importing matplotlib for plotting 

import matplotlib.pyplot as plt



# Get correlations of each features in dataset

corrmat = df_vehicle.corr()

top_corr_features = corrmat.index

plt.figure(figsize = (10, 10))



# Plotting the heat map

g = sns.heatmap(df_vehicle[top_corr_features].corr(), 

                annot = True, 

                cmap = "RdYlGn")
# Seperating dataset for fitting the model

X = final_dataset.iloc[:, 1:]

y = final_dataset.iloc[:, 0]



# Displaying the unique values of total owners of the vehicle

X['Owner'].unique()
# Displaying the top 5 rows of the new dataset

X.head()
y.head()
# Importing the required libraries

from sklearn.ensemble import ExtraTreesRegressor



# Creating an Regressor Model

model = ExtraTreesRegressor()



# Fitting the model

model.fit(X, y)



# Displaying the feature importance array

print(model.feature_importances_)
# Plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index = X.columns)

feat_importances.nlargest(5).plot(kind = 'barh')

plt.show()
# Importing the library to split the data into train and test sets

from sklearn.model_selection import train_test_split



# Beginning the splitting of data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Importing the library for Random Forest Regression

from sklearn.ensemble import RandomForestRegressor



# Creating a Random Forest Regression model

regressor = RandomForestRegressor()
# Importing Randomized Search CV

from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

import numpy as np

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]



# Number of features to consider at every split

max_features = ['auto', 'sqrt']



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]



# max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]



# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



# Printing the Random Grid

print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

# Creating a Random Forest Regression model

rf = RandomForestRegressor()



# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = rf, 

                               param_distributions = random_grid,

                               scoring = 'neg_mean_squared_error', 

                               n_iter = 10, 

                               cv = 5, 

                               verbose = 2, 

                               random_state = 42, 

                               n_jobs = 1)



# Fitting a Random Forest Regression Model

rf_random.fit(X_train,y_train)
# Displaying the best paramenters for the Regression

rf_random.best_params_
# Displaying the best score for the Regression

rf_random.best_score_
# Predicting on the test dataset

predictions = rf_random.predict(X_test)



# Distribution plot for the predictions

sns.distplot(y_test-predictions)
# Scatter plot for the predictions

plt.scatter(y_test, predictions)
# Importing libraries to calculate the metrics

from sklearn import metrics



# Displaying the metrics for the analysis

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#importing required library

#import pickle



#open a file, where you ant to store the data

#file = open('random_forest_regression_model.pkl', 'wb')



#dump information to that file

#pickle.dump(rf_random, file)