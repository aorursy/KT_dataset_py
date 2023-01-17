# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Read the data by csv file
housing_data = pd.read_csv('../input/housing-simple-regression/Housing.csv')
housing_data.head()
housing_data.tail()
#Check the shape of the dataframe
housing_data.shape
#Check the information about the data
housing_data.info()
#Let's check about null values
housing_data.isna().any()
#statstical summary of the data
housing_data.describe()
housing_data.hist(figsize=(20,20))
#let's make pairplot of numerical data
sns.pairplot(housing_data)
plt.show()
#Box plots
plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
sns.boxplot(x='mainroad', y='price', data=housing_data)
plt.subplot(2,3,2)
sns.boxplot(x='guestroom', y='price', data=housing_data)
plt.subplot(2,3,3)
sns.boxplot(x='basement', y='price', data=housing_data)
plt.subplot(2,3,4)
sns.boxplot(x='hotwaterheating', y='price', data=housing_data)
plt.subplot(2,3,5)
sns.boxplot(x='airconditioning', y='price', data=housing_data)
plt.subplot(2,3,6)
sns.boxplot(x='furnishingstatus', y='price', data=housing_data)
plt.show()





#List of variables to map
varlist= ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
#Defining the map function
def binary_map(x):
    return x.map({'yes':1,'no':0})
housing_data[varlist]=housing_data[varlist].apply(binary_map)

housing_data.head()
#Get the dummy variable for the attribute furnishingstatus and store it in a new dataframe
df=pd.get_dummies(housing_data['furnishingstatus'])
#Check how new dataset df looks like
df.head()
#Let's drop first column
df=pd.get_dummies(housing_data['furnishingstatus'], drop_first=True)
df.head()
#Add the above dataframe df into original housing_data dataframe
housing_data=pd.concat([housing_data, df], axis=1)
housing_data.head()
housing_data.drop(['furnishingstatus'], axis=1, inplace= True)
housing_data.head()
#Splitting of data into train and test set
from sklearn.model_selection import train_test_split
train, test = train_test_split(housing_data,train_size=0.8,test_size=0.2,random_state=0)
#Scaling the features
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
#Apply scaler everywhere except categorical data
num = ['area','bedrooms','bathrooms','stories','parking','price']
train[num] = scaler.fit_transform(train[num])
train.head()
#Let's find outh the correlation matrix
corr=housing_data.corr()
corr
#Let's check heatmap
plt.figure(figsize=(16,10))
sns.heatmap(corr,annot=True, cmap='YlGnBu')
plt.show()
from sklearn.linear_model import LinearRegression
X_train= train
Y_train= train.pop('price')

from sklearn.feature_selection import RFE
lm = LinearRegression()
lm.fit(X_train, Y_train)
rfe = RFE(lm, 10)             
rfe = rfe.fit(X_train, Y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
#Display the columns suuported by RFE
col = X_train.columns[rfe.support_]
col
#Display the columns not supported by RFE
X_train.columns[~rfe.support_]
# Creating training dataframe with RFE selected variables
X_train_rfe = X_train[col]
# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)
# Running the linear model
lm = sm.OLS(Y_train,X_train_rfe).fit()   
#Let's see the summary of our linear model
print(lm.summary())
#Drooping the bedrooms column
X_train_new = X_train_rfe.drop(["bedrooms"], axis = 1)
#Rebuilding the model without bedrooms

import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(Y_train,X_train_lm).fit() 




#Let's see the summary of our linear model
print(lm.summary())
X_train_new.columns
X_train_new = X_train_new.drop(['const'], axis=1)
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#Predicted value of price
Y_train_price = lm.predict(X_train_lm)
# Displaying error terms
fig = plt.figure()
sns.distplot((Y_train - Y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                   
plt.xlabel('Errors', fontsize = 18)                         
#Applying the scaling on the test sets
num = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
test[num] = scaler.transform(test[num])
Y_test = test.pop('price')
X_test = test
# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Making predictions
Y_pred = lm.predict(X_test_new)
# Displaying available and predicted price values for the test data 
fig = plt.figure()
plt.scatter(Y_test,Y_pred)
fig.suptitle('Y_test vs Y_pred', fontsize=20)            
plt.xlabel('Y_test', fontsize=18)                          
plt.ylabel('Y_pred', fontsize=16)                          