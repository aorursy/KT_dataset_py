# Import libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os



print(os.listdir("../input"))
# Read the dataset

df = pd.read_csv("../input/insurance.csv")



# Have a first look at the dataset

df.head(5)
# Know the numbers of rows and columns

df.shape
# Know the names of the columns

df.columns
# Get more information about the dataset

# Know the datatypes of the columns

df.info()
# Check if there are missing values

missing_values = (df.isna().sum())

print(missing_values)
# Transform categorical data into numerical data

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()



# Transform the dataype of sex from object to int64

le.fit(df.sex.drop_duplicates()) 

df.sex = le.transform(df.sex)

# Transform the dataype of smoker from object to int64

le.fit(df.smoker.drop_duplicates()) 

df.smoker = le.transform(df.smoker)

# Transform the dataype of region from object to int64

le.fit(df.region.drop_duplicates()) 

df.region = le.transform(df.region)
# Look at statistical details of the data

df.describe()
# Know the correlations between the columns

df.corr()
df.corr()['charges'].sort_values()
# Create features (x) and targets (y)

x = df[['age']]

y = df[['charges']]



print(x.head(5))

print(y.head(5))
# Split the data in training set and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)



print(x_train.shape) 

print(y_train.shape) 

print(x_test.shape)

print(y_test.shape)
# Train the model 

from sklearn.linear_model import LinearRegression 

regr = LinearRegression()

regr = regr.fit(x_train, y_train) 

y_pred = regr.predict(x_test)



# Plot outputs

plt.scatter(x_test, y_test, color='black') 

plt.title('Medical Costs') 

plt.xlabel('Age') 

plt.ylabel('Charges') 

plt.plot(x_test, regr.predict(x_test), color='red',linewidth=3) 

plt.show() 
# Know the coefficient 

print(regr.coef_)
from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  