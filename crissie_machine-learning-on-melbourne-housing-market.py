import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Get the melbourne housing data into DataFrame

df = pd.read_csv('/kaggle/input/melbourne-housing-market/Melbourne_housing_FULL.csv')
# Show first five rows of the data

df.head()
# View the datatypes of columns and how many entries in the DataFrame

df.info()

# View stats of the DataFrame

df.describe()
# View the correlation between columns in data

# In summary, bedrooms column has higher correlation with rooms followed by bathrooms, postcode and price

plt.figure(figsize=(30,20))

sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
# Check for any missing values based on data's columns

df.isnull().sum()
# Create function to fill in missing price

def impute_price(cols):

    Price = cols[0]

    Rooms = cols[1]

    

    if pd.isnull(Price):

        if Rooms == 1:

            return 432889

        elif Rooms == 2:

            return 759484

        elif Rooms == 3:

            return 1028500

        elif Rooms == 4:

            return 1369597

        elif Rooms == 5:

            return 1818862

        elif Rooms == 6:

            return 1882613

        elif Rooms == 7:

            return 1791675

        elif Rooms == 8:

            return 1716858

        elif Rooms == 9:

            return 1380000

        elif Rooms == 10:

            return 2018000

        elif Rooms == 12:

            return 2705000

        elif Rooms == 16:

            return 5000000

        else:

            return 1050173

    else:

        return Price



# Create function to fill in missing number of bedrooms

def impute_bedrooms(cols):

    Bedrooms = cols[0]

    Rooms = cols[1]

    

    if pd.isnull(Bedrooms):

        if Rooms == 1:

            return 1

        elif Rooms == 2:

            return 2

        elif Rooms == 3:

            return 3

        elif Rooms == 4:

            return 4

        elif Rooms == 5:

            return 5

        elif Rooms == 6:

            return 6

        elif Rooms == 7:

            return 7

        elif Rooms == 8:

            return 8

        elif Rooms == 9:

            return 5

        elif Rooms == 10:

            return 10

        elif Rooms == 12:

            return 12

        elif Rooms == 16:

            return 16

        else:

            return 3

    else:

        return Bedrooms
# Create function to fill in missing number of bathrooms

def impute_bathrooms(cols):

    Bathrooms = cols[0]

    Rooms = cols[1]

    

    if pd.isnull(Bathrooms):

        if (Rooms == 1) or (Rooms == 2) or (Rooms == 9):

            return 1

        elif (Rooms == 3) or (Rooms == 4):

            return 2

        elif (Rooms == 5) or (Rooms == 6):

            return 3

        elif (Rooms == 7) or (Rooms == 8):

            return 4

        elif (Rooms == 10) or (Rooms == 12):

            return 5

        elif Rooms == 16:

            return 8

        else:

            return 2

    else:

        return Bathrooms



# Create function to fill in missing number of cars

def impute_cars(cols):

    Cars = cols[0]

    Rooms = cols[1]

    

    if pd.isnull(Cars):

        if (Rooms == 1) or (Rooms == 2):

            return 1

        elif (Rooms == 3) or (Rooms == 4) or (Rooms == 10):

            return 2

        elif (Rooms == 5) or (Rooms == 6) or (Rooms == 12):

            return 3

        elif (Rooms == 7) or (Rooms == 8):

            return 4

        elif Rooms == 9:

            return 5

        elif Rooms == 16:

            return 7

        else:

            return 2

    else:

        return Cars
# Create function to fill in missing building year

def impute_buildyear(cols):

    BuildingYear = cols[0]

    Rooms = cols[1]

    

    if pd.isnull(BuildingYear):

        if (Rooms == 1) or (Rooms == 2) or (Rooms == 3) or (Rooms == 4) or (Rooms == 12):

            return 1998

        elif (Rooms == 5) or (Rooms == 6) or (Rooms == 7) or (Rooms == 8):

            return 1999

        elif Rooms == 10:

            return 2001

        else:

            return 1965

    else:

        return BuildingYear



# Create functions to fill in missing lattitude and longtitude

def impute_lat(cols):

    Lat = cols[0]

    Rooms = cols[1]

    

    if pd.isnull(Lat):

        if Rooms == 1:

            return -37.8241074742268

        elif Rooms == 2:

            return -37.81409220146649

        elif Rooms == 3:

            return -37.80593652024464

        elif Rooms == 4:

            return -37.8120700978892

        elif Rooms == 5:

            return -37.82064511832322

        elif Rooms == 6:

            return -37.80909949720672

        elif Rooms == 7:

            return -37.79428833333334

        elif Rooms == 8:

            return -37.79720117647059

        elif Rooms == 9:

            return -37.84154

        elif Rooms == 10:

            return -37.804758

        elif Rooms == 12:

            return -37.759299999999996

        elif Rooms == 16:

            return -37.81405

        else:

            return -37.810634295599094

    else:

        return Lat

    

def impute_lng(cols):

    Lng = cols[0]

    Rooms = cols[1]

    

    if pd.isnull(Lng):

        if Rooms == 1:

            return 144.98565814432982

        elif Rooms == 2:

            return 144.99202074022355

        elif Rooms == 3:

            return 144.99451968731563

        elif Rooms == 4:

            return 145.01660855919306

        elif Rooms == 5:

            return 145.03943298174434

        elif Rooms == 6:

            return 145.0303535195531

        elif Rooms == 7:

            return 145.04834833333334

        elif Rooms == 8:

            return 145.04010470588236

        elif Rooms == 9:

            return 145.04432333333332

        elif Rooms == 10:

            return 145.06637800000001

        elif Rooms == 12:

            return 144.80714999999998

        elif Rooms == 16:

            return 145.19891

        else:

            return 145.00185113165475

    else:

        return Lng
# The followings apply the functions defined at the top where all columns will be filled in with the correct value

df['Price'] = df[['Price','Rooms']].apply(impute_price,axis=1)



df['Bedroom2'] = df[['Bedroom2','Rooms']].apply(impute_bedrooms,axis=1)

df['Bathroom'] = df[['Bathroom','Rooms']].apply(impute_bathrooms,axis=1)



df['Car'] = df[['Car','Rooms']].apply(impute_cars,axis=1)

df['YearBuilt'] = df[['YearBuilt','Rooms']].apply(impute_buildyear,axis=1)



df['Lattitude'] = df[['Lattitude','Rooms']].apply(impute_lat,axis=1)

df['Longtitude'] = df[['Longtitude','Rooms']].apply(impute_lng,axis=1)
# If you do a box plot for these columns - Landsize and Building Area, you will notice that they have

# no unique mean value to allow some filling in on missing values. Hence, I decide to drop them and

# use the ones we have on the previous cell instead

df.drop(['Landsize','BuildingArea'],axis=1,inplace=True)
# Based on df.isnull().sum(), we can see that Distance, Postcode, CouncilArea, Regionname and Propertycount columns are missing at least two values

# We can go ahead and drop the rows with those empty values because they don't affect the dataset (we have estimated 27,000 entries)

df.dropna(inplace=True)
# Check for any missing values based on data's columns

df.isnull().sum()
# This will be the finalised DataFrame for the machine learning

df
# Importing required modules for machine learning

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score,confusion_matrix,classification_report

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
# Converting building type and sales method columns from finalised DataFrame to dummy categorical variables for machine learning

building_type = pd.get_dummies(df['Type'],drop_first=True)

method = pd.get_dummies(df['Method'],drop_first=True)



# Concatenate building_type and method variables to the DataFrame df

df = pd.concat([df,building_type,method],axis=1)



# Drop the building type and sales method columns from DataFrame as we will use the dummy categorical variables

df.drop(['Type','Method'],axis=1,inplace=True)
# Show first five rows of entries

df.head()
# Dropping any columns that are not integer or float datatype

df.drop(['Address','SellerG','Suburb','Date','CouncilArea','Regionname'],axis=1,inplace=True)
# Preview first five rows of entries

# This will be the finalised DataFrame to be use for splitting dataset and training model for predictions

df.head()
# Interested in testing model for predictions of no. of rooms



# Dropping the rooms column from X

X = df.drop('Rooms',axis=1)



# Holding rooms column in y variable

y = df['Rooms']
# Splitting datset into test and train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Using RandomForestClassifier for example where a RandomForestClassifier model is created

forest = RandomForestClassifier(n_estimators=200)



# Fitting the model with the training dataset

forest.fit(X_train,y_train)
# Go on and let model predict using X_test

predictions = forest.predict(X_test)
# Print out confusion matrix for the model after testing

print(confusion_matrix(y_test,predictions))
# Print out classification report for the model after testing

print(classification_report(y_test,predictions))
# Print out mean squared error, mean absolute error and explained variance score

print('Random Forest Model Stat')

print('Mean Squared Error (MSE): \n' + str(np.sqrt(mean_squared_error(y_test,predictions))) + '\n')

print('Mean Absolute Error (MAE): \n' + str(mean_absolute_error(y_test,predictions)) + '\n')

print('Explained Variance Score: \n' + str(explained_variance_score(y_test,predictions)) + '\n')
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor
tree = DecisionTreeRegressor()

tree.fit(X_train,y_train)
predictions = tree.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print('Decision Tree Regression Model Stat')

print('Mean Squared Error (MSE): \n' + str(np.sqrt(mean_squared_error(y_test,predictions))) + '\n')

print('Mean Absolute Error (MAE): \n' + str(mean_absolute_error(y_test,predictions)) + '\n')

print('Explained Variance Score: \n' + str(explained_variance_score(y_test,predictions)) + '\n')
tree = AdaBoostRegressor(DecisionTreeRegressor(),n_estimators=200)

tree.fit(X_train,y_train)
predictions = tree.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print('Decision Tree Regression with Ada Boost Model Stat')

print('Mean Squared Error (MSE): \n' + str(np.sqrt(mean_squared_error(y_test,predictions))) + '\n')

print('Mean Absolute Error (MAE): \n' + str(mean_absolute_error(y_test,predictions)) + '\n')

print('Explained Variance Score: \n' + str(explained_variance_score(y_test,predictions)) + '\n')
from sklearn.ensemble import GradientBoostingRegressor
grad_model = GradientBoostingRegressor(n_estimators=200)

grad_model.fit(X_train,y_train)
predictions = grad_model.predict(X_test)
print('Gradient Boosting Stat')

print('Mean Squared Error (MSE): \n' + str(np.sqrt(mean_squared_error(y_test,predictions))) + '\n')

print('Mean Absolute Error (MAE): \n' + str(mean_absolute_error(y_test,predictions)) + '\n')

print('Explained Variance Score: \n' + str(explained_variance_score(y_test,predictions)) + '\n')