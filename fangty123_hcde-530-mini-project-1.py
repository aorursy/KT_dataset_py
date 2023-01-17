# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sales = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')



# print the number of data points and types of metadata

print('Number of data points: ' + str(sales.shape[0]))

print('Types of metadata: ' + str(sales.shape[1]))



# sample top 5 lines of sales data to see what's available

# also force pandas to display all columns

pd.set_option('display.max_columns', 110)

sales.head()
filtered_data = sales.loc[:, ['price', 'sqft_living', 'yr_built', 'yr_renovated', 'zipcode', 'bedrooms', 'bathrooms']]

cleaned_data = filtered_data.dropna()



# confirm the shape of cleaned data

print(cleaned_data.shape)



# Next, sample the first 5 rows of data to make sure it contains correct information

cleaned_data.head(5)
# using apply function to create a new column age

cleaned_data['age'] = cleaned_data.apply(lambda row: 1.5 * (2015 - row.yr_renovated) if row.yr_renovated != 0 else (2015 - row.yr_built), axis = 1)



# using apply function to create a new column age

cleaned_data['rooms'] = cleaned_data.apply(lambda row: row.bedrooms + (row.bathrooms * 0.8), axis = 1)



# Next, sample the first 5 rows of data to make sure age and rooms looks correct

cleaned_data.head(5)
import matplotlib.pyplot as plt

plt.figure(figsize=(30, 10))



# first, drop the yr_built, yr_renovated, bedrooms, and bathrooms column as they are no longer needed

final_data = cleaned_data.loc[:, ['price', 'sqft_living', 'age', 'rooms', 'zipcode']]



final_data.plot.scatter(x='sqft_living', y='price', c='DarkBlue')

final_data.plot.scatter(x='age', y='price', c='Blue')

final_data.plot.scatter(x='rooms', y='price', c='Yellow')

final_data.plot.scatter(x='zipcode', y='price', c='Red')
from mpl_toolkits import mplot3d



# create 3D scatter plot with sqft_living and rooms

fig = plt.figure()

sqft_living_age = plt.axes(projection='3d')

zdata = final_data['price']

xdata = final_data['sqft_living']

ydata = final_data['rooms']

sqft_living_age.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Blues');
# sample the data points matching the criteria

sampled_data = final_data.loc[(final_data.sqft_living >= 2800) & (final_data.sqft_living <= 3200) & (final_data.rooms >= 6.0) & (final_data.rooms <= 8.0)] 



# confirm the shape of sampled data

print(sampled_data.shape)



# use bar chart of figure out popular zipcodes

zip_code_data = sampled_data.groupby('zipcode')["price"].count().reset_index(name="count")

# select the top 5 zipcodes after sorting for bar chart

zip_code_data = zip_code_data.sort_values(by='count', ascending=False).head(5)

zip_code_data.plot.bar(x='zipcode', y='count', figsize=(30, 10))



# now lets try each popular zipcode one by one

data_98075 = sampled_data.loc[sampled_data.zipcode == 98075]

data_98075.plot.scatter(x='age', y='price', c='Blue')

data_98059 = sampled_data.loc[sampled_data.zipcode == 98059]

data_98059.plot.scatter(x='age', y='price', c='Blue')

data_98052 = sampled_data.loc[sampled_data.zipcode == 98052]

data_98052.plot.scatter(x='age', y='price', c='Blue')

data_98006 = sampled_data.loc[sampled_data.zipcode == 98006]

data_98006.plot.scatter(x='age', y='price', c='Blue')

data_98038 = sampled_data.loc[sampled_data.zipcode == 98038]

data_98038.plot.scatter(x='age', y='price', c='Blue')

import numpy as np 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression 



x_sqft = np.array(final_data['sqft_living']).reshape(-1, 1)

y_price = np.array(final_data['price'])



# splitting the data for simple regssion based on sqft_living

x_sqft_train, x_sqft_test, y_price_train, y_price_test = train_test_split(x_sqft,y_price,test_size=1/4, random_state=0)



# fitting simple linear regression to the training Set

linear_regressor = LinearRegression()

linear_regressor.fit(x_sqft_train, y_price_train)



# using the linear regression to predict prices in training set

sqft_train_prediction = linear_regressor.predict(x_sqft_train)



# visualizing the prediction line in training set 

plt.scatter(x_sqft_train, y_price_train, color= 'yellow')

plt.plot(x_sqft_train, sqft_train_prediction, color = 'darkblue')

plt.xlabel("sqft_living")

plt.ylabel("price")

plt.show()



# function to calculate the average difference between prediction and actual price

def calculateDiffAvg(y_data, y_prediction):

    print('Length of data set: ' + str(len(y_prediction)))

    sum_diff = 0

    for index in range(len(y_prediction)):

        sum_diff += abs(y_prediction[index] - y_data[index])

    average_diff = sum_diff / len(y_prediction)

    return average_diff



# function to calculate the percentage of consistency

def calculatePercentConsistent(diff_train, diff_test):

    return (1 - (abs(diff_train - diff_test) / diff_train)) * 100



diff_avg_train = calculateDiffAvg(y_price_train, sqft_train_prediction)

print('Average difference in training set: ' + diff_avg_train.astype(str))



# using the linear regression to predict prices in test set

sqft_test_prediction = linear_regressor.predict(x_sqft_test)

diff_avg_test = calculateDiffAvg(y_price_test, sqft_test_prediction)

print('Average difference in test set: ' + diff_avg_test.astype(str))



# calculate the percentage of consistency

per_consist = calculatePercentConsistent(diff_avg_train,diff_avg_test)

print('Percentage of consistency: %s' % per_consist)
x_room = np.array(final_data['rooms']).reshape(-1, 1)



# splitting the data for simple regssion based on rooms

x_room_train, x_room_test, y_price_train, y_price_test = train_test_split(x_room,y_price,test_size=1/4, random_state=0)



# fitting simple linear regression to the training Set

linear_regressor_room = LinearRegression()

linear_regressor_room.fit(x_room_train, y_price_train)



# using the linear regression to predict prices in training set

room_train_prediction = linear_regressor_room.predict(x_room_train)



# visualizing the prediction line in training set 

plt.scatter(x_room_train, y_price_train, color= 'red')

plt.plot(x_room_train, room_train_prediction, color = 'darkblue')

plt.xlabel("rooms")

plt.ylabel("price")

plt.show()
diff_avg_train = calculateDiffAvg(y_price_train, room_train_prediction)

print('Average difference in training set: ' + diff_avg_train.astype(str))



# using the linear regression to predict prices in test set

room_test_prediction = linear_regressor_room.predict(x_room_test)

diff_avg_test = calculateDiffAvg(y_price_test, room_test_prediction)

print('Average difference in test set: ' + diff_avg_test.astype(str))



# calculate the percentage of consistency

per_consist = calculatePercentConsistent(diff_avg_train,diff_avg_test)

print('Percentage of consistency: %s' % per_consist)
# select multiple variables for linear regression this time

x_multi = final_data.loc[:, ['sqft_living', 'rooms', 'age']].values



# splitting the data for simple regssion based on rooms

x_multi_train, x_multi_test, y_price_train, y_price_test = train_test_split(x_multi,y_price,test_size=1/4, random_state=0)



# fitting linear regression to the training Set

linear_regressor_multi = LinearRegression()

linear_regressor_multi.fit(x_multi_train, y_price_train)



# using the linear regression to predict prices in training set

multi_train_prediction = linear_regressor_multi.predict(x_multi_train)



diff_avg_train = calculateDiffAvg(y_price_train, multi_train_prediction)

print('Average difference in training set: ' + diff_avg_train.astype(str))



# using the linear regression to predict prices in test set

multi_test_prediction = linear_regressor_multi.predict(x_multi_test)

diff_avg_test = calculateDiffAvg(y_price_test, multi_test_prediction)

print('Average difference in test set: ' + diff_avg_test.astype(str))



# calculate the percentage of consistency

per_consist = calculatePercentConsistent(diff_avg_train,diff_avg_test)

print('Percentage of consistency: %s' % per_consist)
