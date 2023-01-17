#@title Run this to import libraries and your data! { display-mode: "form" }

#Please run `pip install pandas` in the terminal if the below doesn't work for you

import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv). 

import os # Good for navigating your computer's files 
# read our data in using 'pd.read_csv('file')'

data_path  = '../input/vehicle-dataset-from-cardekho/car data.csv'

car_data = pd.read_csv(data_path)
# let's look at our 'dataframe'. Dataframes are just like google or excel spreadsheets. 

# use the 'head' method to show the first five rows of the table as well as their names. 

car_data.head() 
car_data[['Fuel_Type']].head()
car_data[['Car_Name']].head()
# use the 'len' method to see how many rows are in our dataframe

print(len(car_data))
# first we'll grab our handy visualization tools

import seaborn as sns

import matplotlib.pyplot as plt



# Each dot is a single example (row) from the dataframe, with its 

# x-value as `Year` and its y-value as `Selling_Price`

#To use the  `scatterplot` tool from the `seaborn` plotting package... do the following: 

#sns.scatterplot(x = 'feature_column', y = 'target_column', data = source_data_frame)

sns.scatterplot(x = 'Year', y = 'Selling_Price', data = car_data)
car_data.groupby('Fuel_Type').count()
sns.catplot(x = 'Fuel_Type', y = 'Selling_Price', data = car_data, kind = 'swarm')
sns.scatterplot(x = 'Kms_Driven', y = 'Selling_Price', data = car_data)
sns.catplot(x = 'Seller_Type', y = 'Selling_Price', data = car_data, kind = 'swarm')

sns.catplot(x = 'Transmission', y = 'Selling_Price', data = car_data, kind = 'swarm')
# let's pull our handy linear fitter from our 'prediction' toolbox: sklearn!

from sklearn import linear_model

import numpy as np    # Great for lists (arrays) of numbers



# We're splitting up our data set into groups called 'train' and 'test'

from sklearn.model_selection import train_test_split



x = car_data['Year'].values

x = x[:,np.newaxis]

y = car_data['Selling_Price'].values



# set up our model

linear = linear_model.LinearRegression(fit_intercept = True)



# train the model 

linear.fit(x, y)
#@title Visualize the fit with this cell!

import matplotlib.pyplot as plt



y_pred = linear.predict(x)

plt.plot(x, y_pred, color='red')



plt.scatter(x, y)

plt.xlabel('Year') # set the labels of the x and y axes

plt.ylabel('Selling_Price (lakhs)')

plt.show()
print('Our m is %0.2f lakhs/year'%linear.coef_)
print('Our b is %0.2f lakhs'%linear.intercept_)
m = linear.coef_

b = linear.intercept_

age = 5

selling_price = m * age + b

selling_price
car_data['TransmissionNumber'] = car_data['Transmission'].replace({'Manual':1, 'Automatic':0})
x = car_data[['Year', 'TransmissionNumber', 'Kms_Driven']].values



# set up our model

multiple = linear_model.LinearRegression(fit_intercept = True, normalize = True)



# train the model 

multiple.fit(x, y)
print('Our single linear model had an R^2 of: %0.3f'%linear.score(x[:,[0]], y))
print('Our multiple linear model had an R^2 of: %0.3f'%multiple.score(x, y))
car_data['SellerType_number'] = car_data['Seller_Type'].replace({'Dealer':1, 'Individual':0})

x = car_data[['Year', 'TransmissionNumber', 'Kms_Driven', 'SellerType_number']].values



# set up our model

multiple = linear_model.LinearRegression(fit_intercept = True, normalize = True)



# train the model 

multiple.fit(x, y)



print('Our multiple linear model had an R^2 of: %0.3f'%multiple.score(x, y))