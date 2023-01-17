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
!pip install turicreate

import turicreate as tc
sales = tc.SFrame.read_csv('/kaggle/input/home-prices-dataset/home_data.csv')
# Have a first look at our data 

sales
# Used for quick visualisation and data exploration

sales.show()
# Plot a scatter plot to see the relationship between plot size of living space and the price

tc.visualization.set_target('auto') # to display the graph in the desired location 

tc.visualization.scatter(x=sales["sqft_living"], y=sales["price"], xlabel="Living Area", ylabel="Price", title="Scatter Plot")
tc.show(sales[1:5000]['sqft_living'],sales[1:5000]['price'])
# Splitting the data into training and testing data

training_set, test_set = sales.random_split(.8,seed=0)
# Building our linear regression model

sqft_model = turicreate.linear_regression.create(training_set,target='price',features=['sqft_living'])
# Printing the mean value of the test data

print(test_set['price'].mean())
# Evaluate the model prediction against the actual data for predictions

print(sqft_model.evaluate(test_set))
# Let's look at model weights ( slope and intercept ) that we fit

sqft_model.coefficients
# Visualizing the regression line that was fit in this case 

import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(test_set['sqft_living'],test_set['price'],'.',

        test_set['sqft_living'],sqft_model.predict(test_set),'-')
my_features = ['bedrooms','bathrooms','sqft_living','sqft_basement','floors']
sales[my_features].show()
my_features_model = turicreate.linear_regression.create(training_set,target='price',features=my_features)
print (my_features)
print (sqft_model.evaluate(test_set))

print (my_features_model.evaluate(test_set))
my_features_model.coefficients
# Extracting a particular feature

house1 = sales[sales['id']== 5309101200]
house1
print (house1['price'])
print (sqft_model.predict(house1))
print (my_features_model.predict(house1))
house2 = sales[sales['id']==1925069082]
house2
print(house2['price'])
print (sqft_model.predict(house2))
print (my_features_model.predict(house2))
bill_gates = {'bedrooms':[8], 

              'bathrooms':[25], 

              'sqft_living':[50000], 

              'sqft_lot':[225000],

              'floors':[4], 

              'zipcode':['98039'], 

              'condition':[10], 

              'grade':[10],

              'waterfront':[1],

              'view':[4],

              'sqft_above':[37500],

              'sqft_basement':[12500],

              'yr_built':[1994],

              'yr_renovated':[2010],

              'lat':[47.627606],

              'long':[-122.242054],

              'sqft_living15':[5000],

              'sqft_lot15':[40000]}
print (my_features_model.predict(turicreate.SFrame(bill_gates)))