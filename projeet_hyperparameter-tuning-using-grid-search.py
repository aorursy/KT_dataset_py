# Importing Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
plt.style.use('seaborn-dark')

import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Loading the dataset
car_data = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")
# Head
car_data.head()
# Car Details
car_details = pd.read_csv("../input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv")
car_details.head()
# Checking the missing values and datatypes of the features
car_data.info()
# Checking the distribution of the data
car_data.describe()
car_data['Car_Name'].value_counts()
# Dropping car name column
car_data.drop(['Car_Name'], axis = 1, inplace = True)
# Saving the original data
data = car_data.copy()
data.head()
# Creating a new columns with the age of the car
data['car_age'] = 2020 - data['Year']
# Dropping the year column
data.drop('Year', axis =1, inplace = True)
# Scatterplot to see the trend of selling price over the years
sns.scatterplot(x = data['car_age'], y = data['Selling_Price'], data = data)
# Scatterplot to see the impact of the 'Kms_Driven' on the sale of the car 
sns.scatterplot(x = data['Kms_Driven'], y = data['Selling_Price'], data = data)
# Barplot to see how 'Seller_Type' affects the 'Selling_Price'
sns.barplot(x = data['Seller_Type'], y = data['Selling_Price'])
# Barplot to see how 'Transmission' affects the 'Selling_Price'
sns.barplot(x = data['Transmission'], y = data['Selling_Price'])
# Barplot to see how 'Owner' affects the 'Selling_Price'
sns.barplot(x = data['Owner'], y = data['Selling_Price'])
plt.figure(figsize=(12,10))
sns.jointplot(x='Present_Price',y='Selling_Price',data=data)
plt.title('Present_Price vs Selling_Price',fontweight="bold", size=20)
plt.show()
plt.figure(figsize=(12,10))
sns.jointplot(x='Kms_Driven',y='Selling_Price',data=data, kind = 'hex')
plt.title('Kms_Driven vs Selling_Price',fontweight="bold", size=20)
plt.show()
data.head()
plt.figure(figsize = (10,4))
plt.subplot(1,2,1)
sns.violinplot(x = 'Fuel_Type',y ='Selling_Price', data = data)
plt.subplot(1,2,2)
sns.violinplot(x = 'Transmission',y ='Selling_Price', data = data)
data.head()
# Dummy Encoding

fuel_type = pd.get_dummies(data['Fuel_Type'], drop_first= True)
seller_type = pd.get_dummies(data['Seller_Type'], drop_first= True)
transmission = pd.get_dummies(data['Transmission'], drop_first= True)

data = pd.concat([data,fuel_type,seller_type,transmission], axis = 1)

data.head()
# Dropping the dummified columns

data.drop(['Fuel_Type','Seller_Type','Transmission'], axis = 1, inplace = True)
data.head()
from sklearn.tree import DecisionTreeRegressor
# Setting the max_depth at 5
dt = DecisionTreeRegressor(random_state=42)
np.random.seed(0)
df_train, df_test = train_test_split(data, train_size=0.8, random_state=100)
df_train.shape, df_test.shape
y_train = df_train.pop("Selling_Price")
X_train = df_train

y_test = df_test.pop("Selling_Price")
X_test = df_test
# Fitting the decision tree
dt.fit(X_train, y_train)
# Visualization of the decision tree
from sklearn.tree import plot_tree
plt.figure(figsize=(40,12))
plot_tree(dt, feature_names = df_train.columns,filled=True);
# Making Prediction on training data set
y_train_pred = dt.predict(X_train)
# Clearly model has over-fitted the data
r2_score(y_train, y_train_pred)
# Making Prediction on test data set
y_test_pred = dt.predict(X_test)
# Not a very good score as compared to the training data r2 score
r2_score(y_test, y_test_pred)
dt = DecisionTreeRegressor(random_state = 42)
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [11,13,15,17,19],
    'min_samples_leaf': [1,3,5,7],
    'criterion': ["mse","friedman_mse", "mae"]
}
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = 'r2')
%%time
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()
score_df.nlargest(5,"mean_test_score")
grid_search.best_estimator_
dt_best = grid_search.best_estimator_
def evaluate_model(dt_regressor):
    print("Train r2 :", r2_score(y_train, dt_regressor.predict(X_train)))
    
    
    print("-"*50)
    print("Test r2 :", r2_score(y_test, dt_regressor.predict(X_test)))
   
evaluate_model(dt_best)
# Importing the Random Forest Library
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=11, min_samples_leaf=3)
# Fitting the model into training data
rf.fit(X_train, y_train)
sample_tree = rf.estimators_[30]
sample_tree
# Making predictions for training and test data
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
# R2 score on the training data
r2_score(y_train, y_train_pred)
# R2 score on test data
r2_score(y_test, y_test_pred)
# Create the parameter grid based on the results of random search 
n_estimators = np.arange(100,200,10)
params = {
    'n_estimators': n_estimators,
    'max_depth': [11,13,15,17,19],
    'min_samples_leaf': [1,3,5,7],
    'criterion': ["mse","friedman_mse", "mae"]
}
rf = RandomForestRegressor(random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = 'r2')
%%time
grid_search.fit(X_train, y_train)
grid_search.best_estimator_
def evaluate_model(rf_regressor):
    print("Train r2 :", r2_score(y_train, rf_regressor.predict(X_train)))
    
    
    print("-"*50)
    print("Test r2 :", r2_score(y_test, rf_regressor.predict(X_test)))
# Best Estimator
rf_best = grid_search.best_estimator_
# Getting a better r2 score of 0.83
evaluate_model(rf_best)
