# Import the required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import LabelEncoder

import math

%matplotlib inline
# load dataset to pandas data frame

data = pd.read_csv("./google-play-store-apps/googleplaystore.csv")
# Checking first 5 rows to get overview of the data

data.head()
data.shape
# Checking the data type of the columns

data.info()
# Exploring missing data and checking which all columns have NaN values

data.isnull().any()
# Exploring missing data and checking how many NaN values we have for each column

data.isnull().sum()
# Exploring missing data in heatmap format

plt.figure(figsize = (10,5))

sns.heatmap(data.isnull())
# to get the shape of original data set

data.shape
# removing null values from the data set

data.dropna(inplace = True)
# to get the shape of  data set after dropping the rows which contains NaN values

data.shape
# Checking whether still NaN values present in the data set

data.isnull().any()
# Getting all the column names available in data set

data.columns
# Visualizing how the category of App impacting the rating

plt.figure(figsize = (5,7))

sns.barplot(y = data.Category, x =  data.Rating)

plt.title("Avg rating of Apps based on Category")
plt.figure(figsize=(12, 5))

# Checking how many Apps are free or paid

plt.subplot(1,2,1)

sns.countplot(x='Type',data=data)

plt.title("No of free and paid Google Apps")

# Checking how type of Apps impacting the rating of App

plt.subplot(1,2,2)

sns.barplot(x='Type', y='Rating', data=data)

plt.title("Avg rating based on Type of App")
plt.figure(figsize=(12, 5))

sns.barplot(x='Content Rating', y='Rating',  data = data)

plt.title("Avg Rating based on content Rating of App ")

plt.show
plt.figure(figsize=(5,20))

sns.barplot(y='Genres', x='Rating',  data = data)

plt.title("Avg App Rating based on Genres of App")

plt.show
plt.figure(figsize=(5,100))

sns.barplot(x = 'Rating', y = 'Size', data = data)

plt.title("Avg App Rating based on Size of App")
# Creating class object to Encode the categorical data

le = LabelEncoder()
#Genres Feature Encoding using LabelEncoder

data['App'] = le.fit_transform(data['App'])
# Content Rating Feature encoding using labelEncoder

data['Content Rating'] = le.fit_transform(data['Content Rating'])

# Genres Feature encoding using labelEncoder

data['Genres'] = le.fit_transform(data['Genres'])
# Type Feature encoding using labelEncoder

data['Type'] = pd.get_dummies(data['Type'])
# Creating Dummies for Category features with Dummies function in Pandas

data = pd.concat([data, pd.get_dummies(data['Category'], prefix='cat', drop_first = True)], axis=1)
# Cleaning the Size columns by converting all values to MB and removing the M from the value

data['Size'] = data['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

data['Size'] = data['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)

data['Size'] = data['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)

# Replace Varies with device  size with 0

data[data['Size'] == 'Varies with device']  = 0

# Converting the dtype of Size column value to Float dtype

data['Size'] =  data['Size'].apply(lambda x: float(x))
# Cleaning the Installs column values by removing "+", "," from the Installs Column

data['Installs'] = data['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else x)

data['Installs'] = data['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else x)

# Converting the dtype of Installs column value to Float dtype

data['Installs'] = data['Installs'].apply(lambda x: float(x))
# Cleaning the Price column values by removing "$" from the Price Column

data['Price'] =  data['Price'].apply(lambda x: str(x).strip('$') )

# Converting the dtype of Price columns value to Float dtype

data['Price']= data['Price'].apply(lambda x: float(x))
# checking the coulumns after Encoding the Categorical data

data.columns
# dropping columns which are not required/ doesn't have have any impact on predicting

data.drop(['Last Updated', 'Current Ver','Android Ver', 'Category'] , axis = 1, inplace = True)
#creating features dataset X and target dataset y

X = data.drop(['Rating'], axis = 1)

y = data.Rating
# spliting data into testing and train set 



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 324)
#creating Linear Regression object

linear_model = LinearRegression()
# training the Linear Regression Model with training data set X_train, Y_train



linear_model.fit(X_train, y_train)
# Calculate accuracy of Linear Regression model

accuracy_Linear = linear_model.score(X_test, y_test)

accuracy_Linear
# Predicting the Rating of Apps using Linear model created

linear_regression_prediction = linear_model.predict(X_test)

# Caluationg Root Mean Squared Error of Linear Regression Model

RMSE_linear =  np.sqrt(mean_squared_error(y_test, linear_regression_prediction))

RMSE_linear
# Calculating Mean Absolute Error of Linear Regression Model

MAE_linear =  (mean_absolute_error(y_test, linear_regression_prediction))

MAE_linear
# creating Decision Tree Regressor model object

decision_tree_regressor = DecisionTreeRegressor()
# creating a function which will return the best max depth value

def best_max_depth(n):

    max_depth = np.arange(1, n , 2)

    dscores = []

    for n in max_depth:

        decision_tree_regressor.set_params(max_depth = n)

        decision_tree_regressor.fit(X_train, y_train)

        dscores.append(decision_tree_regressor.score(X_test, y_test))

#     plt.figure(figsize=(7, 5))

#     plt.title("Effect of Estimators")

#     plt.xlabel("Number of Neighbors K")

#     plt.ylabel("Score")

#     plt.plot(n_neighbors, nscores)

    max_depth_data = pd.concat([pd.DataFrame(max_depth), pd.DataFrame(dscores)], axis = 1)

    max_depth_data.columns = ['max_depth', 'dscores']

    ddf = max_depth_data[max_depth_data['dscores' ] == max_depth_data['dscores' ].max() ]



    return ddf.iloc[0, 0]

    
# provide the number to check best max_depth value between 1 to number provided

dn = 30

best_depth = best_max_depth(dn) 

decision_tree_regressor.set_params(max_depth= best_depth)
# training the Decision Tree Regressor model with training data set X_train, Y_train

decision_tree_regressor.fit(X_train, y_train)
# Calculate accuracy of Decision Tree Regressor model

accuracy_decision = decision_tree_regressor.score(X_test,y_test)

accuracy_decision
# Predicting the Rating of Apps using Decision Tree Regressor model created

decision_tree_prediction = decision_tree_regressor.predict(X_test)
# Calculating Mean Absolute Error of Decision Tree Regressor model

RMSE_decision =  np.sqrt(mean_squared_error(y_test, decision_tree_prediction))

RMSE_decision
# Calculating Mean Absolute Error of Decision Tree Regressor model

MAE_decision =  mean_absolute_error(y_test, decision_tree_prediction)

MAE_decision
# creating Knearest Nieghbour regressor model object

KNN_regressor = KNeighborsRegressor()
# creating a function which will return the best max depth value

def best_n_neighbor(n):

    n_neighbors = np.arange(1, n, 1)

    nscores = []

    for n in n_neighbors:

        KNN_regressor.set_params(n_neighbors=n)

        KNN_regressor.fit(X_train, y_train)

        nscores.append(KNN_regressor.score(X_test, y_test))

#     plt.figure(figsize=(7, 5))

#     plt.title("Effect of Estimators")

#     plt.xlabel("Number of Neighbors K")

#     plt.ylabel("Score")

#     plt.plot(n_neighbors, nscores)

    n_neighbors_data = pd.concat([pd.DataFrame(n_neighbors), pd.DataFrame(nscores)], axis = 1)

    n_neighbors_data.columns = ['n_neighbors', 'nscores']

    ndf = n_neighbors_data[n_neighbors_data['nscores' ] == n_neighbors_data['nscores' ].max() ]



    return ndf.iloc[0, 0]

    
# provide the number to check best n_nieghbor value between 1 to number provide

number = 20

n = best_n_neighbor(number) 

KNN_regressor.set_params(n_neighbors= n)
# training the Knearest Nieghbour regressor Model with training data set X_train, Y_train

KNN_regressor.fit(X_train, y_train)
# Calculate mean accuracy of Knearest Nieghbour regressor Model

accuracy_KNN = KNN_regressor.score(X_test, y_test)

accuracy_KNN
# Predicting the Rating of Apps using Knearest Nieghbour regressor Model created

KNN_prediction = KNN_regressor.predict(X_test)
# Calculating Mean Absolute Error of Knearest Nieghbour regressor Model

RMSE_KNN =  np.sqrt(mean_squared_error(y_test, KNN_prediction))

RMSE_KNN
# Calculating Mean Absolute Error of Knearest Nieghbour regressor Model

MAE_KNN =  mean_absolute_error(y_test, KNN_prediction)

MAE_KNN
# creating Random Forest Regressor model object

Random_forest_regressor = RandomForestRegressor()
# creating a function which will return the best n_estimator value

def best_n_estimator(n):

    n_estimator =  np.arange(1,n,5)

    rscores = []

    for n in n_estimator:

        Random_forest_regressor.set_params(n_estimators = n)

        Random_forest_regressor.fit(X_train, y_train)

        rscores.append(Random_forest_regressor.score(X_test, y_test))

#     plt.figure(figsize=(7, 5))

#     plt.title("Effect of n_estimator")

#     plt.xlabel("no. n_estimator")

#     plt.ylabel("score")

#     plt.plot(n_estimator, scores)

    n_estimator_data = pd.concat([pd.DataFrame(n_estimator), pd.DataFrame(rscores)], axis = 1)

    n_estimator_data.columns = ['n_estimator', 'rscores']

    rdf = n_estimator_data[n_estimator_data['rscores' ] == n_estimator_data['rscores' ].max() ]



    return rdf.iloc[0, 0]

    
# provide the number to check best n_estimator value between 1 to number provided

nj = best_n_estimator(120) 

Random_forest_regressor.set_params(n_estimators= nj)
# training the Random Forest Regressor model with training data set X_train, Y_train

Random_forest_regressor.fit(X_train, y_train)
# Calculate mean accuracy of Random Forest Regressor model

accuracy_RFR = Random_forest_regressor.score(X_test, y_test)

accuracy_RFR
# Predicting the Rating of Apps using Random Forest Regressor model created

RFR_prediction = KNN_regressor.predict(X_test)
# Calculating Mean Absolute Error of Random Forest Regressor model

RMSE_RFR =  np.sqrt(mean_squared_error(y_test, RFR_prediction))

RMSE_RFR
# Calculating Mean Absolute Error of Random Forest Regressor model

MAE_RFR =  mean_absolute_error(y_test, RFR_prediction)

MAE_RFR
# Preparing the data frame with the values of the Accuracy score, Root Mean Squared Error and Mean Absolute Error

model_data = pd.DataFrame([[accuracy_Linear,RMSE_linear,MAE_linear], 

                           [accuracy_decision,RMSE_decision,MAE_decision],

                            [accuracy_KNN,RMSE_KNN,MAE_KNN],

                              [accuracy_RFR,RMSE_RFR,MAE_RFR]])
# labelling columns

model_data.columns = ['Accuracy_Score','Root_Mean_Squared_Error','Absolute_Squared_Error ']
# labelling Rows

model_data.index = ['Linear Regression', 'Decision tree regressor','KNN regressor', 'Random forest regressor']
model_data
# Ploting the accuracy score values by sorting them

fig= plt.figure(figsize=(8,6))

model_data['Accuracy_Score'].sort_values().plot.barh()

plt.title("Model Accuracy Score")
