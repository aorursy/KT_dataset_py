import pandas as pd

test_data = pd.read_csv("../input/testset.csv")

training_data = pd.read_csv("../input/trainingset.csv")

training_data.head()

#Author Andres Hernandez Course: cosc3000

training_data.info()
test_data['basement_present'] = test_data['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) # Indicate whether there is a basement or not

test_data['renovated'] = test_data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # 1 if the house has been renovated

training_data['basement_present'] = training_data['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) # Indicate whether there is a basement or not

training_data['renovated'] = training_data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # 1 if the house has been renovated

training_data['living_squared']=training_data['sqft_living']*training_data['sqft_living']

test_data['living_squared']=test_data['sqft_living']*test_data['sqft_living']

training_data.columns

training_data.head()
feature_cols_1 = [ u'bedrooms', u'bathrooms', u'sqft_living',

       u'sqft_lot', u'floors', u'waterfront', u'view', u'condition', u'grade',

       u'sqft_above', u'sqft_basement', u'yr_built', u'yr_renovated', u'zipcode',

       u'lat', u'long', u'sqft_living15', u'sqft_lot15']

feature_cols_2 = [ u'bedrooms', u'bathrooms', u'sqft_living',

       u'sqft_lot', u'floors', u'waterfront', u'view', u'condition', u'grade',

       u'sqft_above', u'yr_built', u'zipcode',

       u'lat', u'long', u'sqft_living15', u'sqft_lot15',

       u'renovated']

#BSA

feature_cols_3 = [ u'bedrooms', u'bathrooms', u'sqft_living',

       u'sqft_lot', u'floors', u'waterfront', u'view', u'condition', u'grade',

       u'sqft_above', u'yr_built', u'zipcode', u'yr_renovated',

       u'lat', u'long', u'sqft_living15', u'sqft_lot15', u'sqft_living', u'living_squared']

feature_cols_4= [ u'bedrooms', u'bathrooms', u'sqft_living',

       u'sqft_lot', u'waterfront', u'view', u'condition', u'grade',

       u'sqft_above', u'yr_built', u'zipcode', u'yr_renovated',

       u'lat', u'long', u'sqft_living15', u'sqft_lot15', u'living_squared']

feature_cols_5 = [ u'bedrooms', u'bathrooms', u'sqft_living',

       u'waterfront', u'view', u'condition', u'grade',

       u'sqft_above', u'yr_built', u'zipcode', u'yr_renovated',

       u'lat', u'long', u'sqft_living15', u'sqft_lot15']

feature_cols_6 = [ u'bedrooms', u'bathrooms', u'sqft_living',

       u'waterfront', u'view', u'condition', u'grade',

       u'sqft_above', u'yr_built', u'zipcode',

       u'lat', u'long', u'sqft_living15', u'sqft_lot15']

x1 = training_data[feature_cols_1]

x2 = training_data[feature_cols_2]

x3 = training_data[feature_cols_3]

x4 = training_data[feature_cols_4]

x5 = training_data[feature_cols_5]

x6 = training_data[feature_cols_6]

y = training_data["price"]



t1 = test_data[feature_cols_1]

t2 = test_data[feature_cols_2]

t3 = test_data[feature_cols_3]

t4 = test_data[feature_cols_4]

t5 = test_data[feature_cols_5]

t6 = test_data[feature_cols_6]

yt = test_data["price"]
from sklearn.linear_model import LinearRegression

regressor1 = LinearRegression()

regressor1.fit(x1, y)

regressor2 = LinearRegression()

regressor2.fit(x2, y)

regressor3 = LinearRegression()

regressor3.fit(x3, y)

regressor4 = LinearRegression()

regressor4.fit(x4, y)

regressor5 = LinearRegression()

regressor5.fit(x5, y)

regressor6 = LinearRegression()

regressor6.fit(x6, y)

accuracy = regressor1.score(t1, yt)

"Accuracy: {}%".format(accuracy * 100)

accuracy = regressor2.score(x2, y)

"Accuracy: {}%".format(accuracy * 100)
accuracy = regressor3.score(x3, y)

"Accuracy: {}%".format(accuracy * 100)
accuracy = regressor4.score(t4, yt)

"Accuracy: {}%".format(accuracy * 100)
accuracy = regressor5.score(t5, yt)

"Accuracy: {}%".format(accuracy * 100)
accuracy = regressor6.score(t6, yt)

"Accuracy: {}%".format(accuracy * 100)
feature_cols_7 = [ u'sqft_living']

x7 = training_data[feature_cols_7]

t7 = test_data[feature_cols_7]

regressor7 = LinearRegression()

regressor7.fit(x7, y)

accuracy = regressor7.score(t7, yt)

"Accuracy: {}%".format(accuracy * 100)
feature_cols_8 = [ u'bedrooms', u'bathrooms', u'sqft_living',

       u'waterfront', u'view', u'condition', u'grade',

       u'sqft_above', u'yr_built', u'zipcode',

       u'lat', u'long']

x8 = training_data[feature_cols_8]

t8 = test_data[feature_cols_8]

regressor8 = LinearRegression()

regressor8.fit(x8, y)

accuracy = regressor8.score(t8, yt)

"Accuracy: {}%".format(accuracy * 100)

cars=regressor8.get_params(True)

for x in cars:

    print (x)

    for y in cars[x]:

        print (y,':',cars[x][y])
#5 is still 70%

feature_cols_9 = [ u'sqft_living',

            u'grade', u'sqft_above']

x9 = training_data[feature_cols_9]

t9 = test_data[feature_cols_9]

regressor9 = LinearRegression()

regressor9.fit(x9, y)

accuracy = regressor9.score(t9, yt)

"Accuracy: {}%".format(accuracy * 100)
import matplotlib.pyplot as plt

x = price