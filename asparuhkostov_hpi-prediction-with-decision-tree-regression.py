################################# IMPORTING ALL MODULES 

import numpy as np #numpy

import pandas as pd #pandas 

from sklearn.tree import DecisionTreeRegressor #scikit-learn's decision tree regression model

from sklearn.metrics import mean_absolute_error #scikit-learn's mean absolute error functions

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt #matplotlib

#this line tells matlotlib to display any plots within the jupyter notebook cells

%matplotlib inline 
################################# DATA LOADING, CLEANING 

df = pd.read_csv('kc_house_data.csv')#creates a dataframe from the dataset

filtered_df = df.dropna(axis=0) #filters missing values

################################# PREDICTION TARGETS + PREDICTORS

prediction_target = filtered_df.price #selects the price column as the column we want to predict

predictors_list = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'bathrooms'] #a list of columns that will be used to predict the price

predictor_data = filtered_df[predictors_list] #creates a df with the predictor data
################################# DECISION TREE REGRESSION MODEL

kingCountyModel = DecisionTreeRegressor() #create a model instance

kingCountyModel.fit(predictor_data, prediction_target) #fit the data within the model instance
################################# QUICK EXERCISE TO CHECK IF THE MODEL IS WORKING

print('Original data used as predictor_data:')

print(predictor_data.head()) #prints the original predictor data's first 5 rows

print('Predicted data, using predictor_data and prediction target:')

kingCountyModel.predict(predictor_data.head()) #prints predicted prices based on the data above
################################# MODEL EVALUATION

#calculating MAE(mean absolute error) with in-sample info

print('In-Sample MAE:')

kingCountyModel = DecisionTreeRegressor()

kingCountyModel.fit(predictor_data,prediction_target)

predicted_data = kingCountyModel.predict(predictor_data) #predict data to use for the MAE calculation

mean_absolute_error(prediction_target, predicted_data) #check the prediction error for each house by using the formula error = actual price − predicted price
################################# MODEL EVALUATION

#using the train_test_split function create 4 different sample dataframes in order to be able to compare out-of-sample values when we calculate the MAE

print('Out-Of-Sample MAE:')

predictor_data_train, predictor_data_val, prediction_target_train, prediction_target_val = train_test_split(predictor_data, prediction_target, random_state = 0)

new_model = DecisionTreeRegressor()

new_model.fit(predictor_data_train, prediction_target_train)

new_prediction = new_model.predict(predictor_data_val)

mean_absolute_error(prediction_target_val, new_prediction)
################################## PLOTTING THE ORIGINAL AND THE PREDICTED PRICES FOR VISUAL COMPARISON

plt.plot(new_prediction[:21], '--', label='Predicted prices', linewidth=2)

plt.plot(prediction_target[:21], '--', label='Original prices', linewidth=2)

plt.legend()

plt.xlabel('N. of Houses')

plt.ylabel('Price')

plt.show()