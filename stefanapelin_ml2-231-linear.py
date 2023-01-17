#Importing the libraries
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
#Step 2: Importing the dataset
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.dtypes
#Taking care of Missing Data in Dataset
#Checking whether we have any null values in our dataset
df.isnull().any()
#defining the attributes and labels
X = df[['price', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat', 'long','sqft_living15','sqft_lot15']]
y = df['sqft_living']
#Step 5
#splitting the data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#testing out sqft_above
print(X_train.shape)
print(X_test.head())
print(y_train.head())
print(y_test.shape)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#because this is a multivariable linear regression, i am checking out the optimal coefficients that the model has chosen
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df
#Checking the R2 for the model
r_sq = regressor.score(X_train, y_train)
print('Coeffiencient of determination(R2):', r_sq)
print('Intercept:', regressor.intercept_)
print('Slope: ', regressor.coef_)
y_pred = regressor.predict(X_test)
y_pred
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head(25)
df.head(25).plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))