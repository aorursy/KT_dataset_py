# Import libraries and read data
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('dataset3_with_missing.csv')
data.info()
# Exploring the values
for col in data.columns[2:-2]:
    print('Mean of ', col, '=', np.mean(data[col]))
    print('Values of ', col, '=', data[col].unique(), '\n')
# List the missing values in each column
data.isnull().sum()
# Check if there are missing values in the columns in the same row
data[data.sqft_above.isnull() & data.sqft_basement.isnull()].shape
# Impute Missing values with fomula: sqft_living = sqft_above + sqft_basement
for row in data.loc[data.sqft_living.isnull()].index:
    data.loc[row, 'sqft_living'] = data.loc[row, 'sqft_above'] + data.loc[row, 'sqft_basement']
    
for row in data.loc[data.sqft_above.isnull()].index:
    data.loc[row, 'sqft_above'] = data.loc[row, 'sqft_living'] - data.loc[row, 'sqft_basement']
    
for row in data.loc[data.sqft_basement.isnull()].index:
    data.loc[row, 'sqft_basement'] = data.loc[row, 'sqft_living'] - data.loc[row, 'sqft_above']
# Create 2 dataframes for training and predicting
df_impute = data.copy()
df_drop_na = data.copy()
df_drop_na.dropna(subset=['bathrooms'],axis=0,inplace=True)
# Compare the NaN values in the 2 dataframes
print(df_drop_na.isnull().sum())
print(df_impute.isnull().sum())
# Plot a scatter matric to find correclations 
pd.plotting.scatter_matrix(df_drop_na.iloc[:,2:-5], alpha = 0.3, figsize = (20,20), diagonal='kde');
plt.show()
# List the correlations for each column 
df_drop_na.corr()["bathrooms"].sort_values()
# Create w dataframes for training and testing
df_drop_na = df_drop_na[['id','date','yr_renovated','zipcode','lat','long',
                     'waterfront','view','condition','sqft_lot','sqft_basement',
                     'price','grade','yr_built','bedrooms','floors','sqft_living','sqft_above','bathrooms']]

df_impute = df_impute[['id','date','yr_renovated','zipcode','lat','long',
                     'waterfront','view','condition','sqft_lot','sqft_basement',
                     'price','grade','yr_built','bedrooms','floors','sqft_living','sqft_above','bathrooms']]
# List the selected predictors
df_drop_na.iloc[:,11:-1]
# Create a linear regression model
X = df_drop_na.iloc[:,11:-1]
y = df_drop_na.iloc[:,-1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  
print ('r-squared: = ', regressor.score(X_test,y_test))
# Predict the missing values
predic = regressor.predict(df_impute[df_impute['bathrooms'].isna()].iloc[:,11:-1])
for row, index in enumerate(df_impute[df_impute['bathrooms'].isnull()].index):
    df_impute.loc[index, 'bathrooms'] = predic[row]
# Check if there is any missing value after imputing data
df_impute['bathrooms'].isnull().unique()
# Fit the model and test the R^2 scores 
X = df_impute.iloc[:,11:-1]
y = df_impute.iloc[:,-1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  
print ('r-squared: = ',regressor.score(X_test,y_test))
# Compare the predicted values with full dataset
boxplot = pd.DataFrame({'Predicted Data': df_impute['bathrooms'], 'Full Data': data['bathrooms']})
boxplot.plot(kind='box', figsize=(10,10))
df_impute.to_csv('dataset3_solution.csv')