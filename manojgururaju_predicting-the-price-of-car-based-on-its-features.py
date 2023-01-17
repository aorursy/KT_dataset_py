# importing the libraries for CarPrice_Assignment

# packages for operations on data
import numpy as np
import pandas as pd
# packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Libraries for scaling and spliting data set
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Libraries for selectong feature and Modeling
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# Librarues for modeling and getting the stats information of the model
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# reading data

car_data = pd.read_csv("../input/header/CarPrice_Assignment.csv")
car_data.head()
# data exploration

print(car_data.shape)

print(car_data.info())

print(car_data.describe())
# extracting car name from car model name

car_data['CarName'] = car_data['CarName'].str.split(' ').str[0] 
car_data['CarName'].head(10)
# verfying the car names

car_data.CarName.unique()
# data cleaning

car_data.CarName = car_data.CarName.replace('maxda', 'mazda')
car_data.CarName = car_data.CarName.replace('Nissan', 'nissan')
car_data.CarName = car_data.CarName.replace('porcshce', 'porsche')
car_data.CarName = car_data.CarName.replace('toyouta', 'toyota')
car_data.CarName = car_data.CarName.replace('vokswagen', 'volkswagen')
car_data.CarName = car_data.CarName.replace('vw', 'volkswagen')
car_data.CarName.unique()
# price distribution

plt.title("car price distribution")
sns.distplot(car_data.price)
plt.show()
plt.figure(figsize=(20, 7))

plt.subplot(1,2,1)
plt.title("Car vs price")
sns.barplot(x='CarName', y='price', data=car_data)
plt.xticks(rotation=45)


plt.subplot(1,2,2)
plt.title("Most purchased cars")
sns.countplot(x='CarName', data=car_data)
plt.xticks(rotation=45)
plt.show()

plt.show()
# factors deciding the price 

plt.figure(figsize=(20, 7))

plt.subplot(1,3,1)
plt.title("cartype vs price")
sns.barplot(x='carbody', y='price', data=car_data)
plt.xticks(rotation=45)

plt.subplot(1,3,2)
plt.title("fueltype vs price")
sns.barplot(x='fueltype', y='price', data=car_data)
plt.xticks(rotation=45)

plt.subplot(1,3,3)
plt.title("cylindernumber vs price")
sns.barplot(x='cylindernumber', y='price', data=car_data)
plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(20, 7))
plt.title("horsepower vs price")
sns.barplot(x='horsepower', y='price', data=car_data)
plt.xticks(rotation=45)
plt.show()
# Correlation between the variables

plt.figure(figsize=(20, 7))
sns.heatmap(car_data.corr(), annot=True)
plt.show()
car_data['FuelEconomy'] = (0.50 * car_data['citympg']) + (0.50 * car_data['highwaympg'])
car_data.head()
car_data.columns.tolist()
# Selecting the features based on the correlations and initial assumption

car_lr = car_data[['price','wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'carheight', 'boreratio', 'horsepower', 'FuelEconomy', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', ]]
car_lr
# visualizing the data set using pairplots

sns.pairplot(car_lr)
plt.show()
# encoding the categorical data

def get_dummies_drop_original(x, df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

car_lr = get_dummies_drop_original('fueltype', car_lr)
car_lr = get_dummies_drop_original('aspiration', car_lr)
car_lr = get_dummies_drop_original('carbody', car_lr)
car_lr = get_dummies_drop_original('enginelocation', car_lr)
car_lr = get_dummies_drop_original('enginetype', car_lr)
car_lr = get_dummies_drop_original('fuelsystem', car_lr)
car_lr = get_dummies_drop_original('drivewheel', car_lr)
car_lr = get_dummies_drop_original('doornumber', car_lr)
car_lr = get_dummies_drop_original('cylindernumber', car_lr)

car_lr
# data exploration

print(car_lr.shape)

print(car_lr.describe())
# splitting the test and train set from the data

df_train, df_test = train_test_split(car_lr, train_size = 0.7, test_size = 0.3, random_state = 100)
# re-scaling

numeric_columns = ['price','wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'carheight', 'boreratio', 'horsepower', 'FuelEconomy']

scaler = MinMaxScaler()

df_train[numeric_columns] = scaler.fit_transform(df_train[numeric_columns])

df_train
# data exploration

print(df_train.shape)

print(df_train.describe())
# spliting the data into X and y variables

y_train = df_train.pop('price')
X_train = df_train
# data exploration

print(y_train.shape)

print(X_train.shape)

print(X_train.columns)
# modelling

lm = LinearRegression()
lm.fit(X_train, y_train)
# selecting the features

rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)
# ranking the features

list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# columns selected for modelling

col = X_train.columns[rfe.support_]
col
# columns rejected

X_train.columns[~rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe.head()
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)
# Running the linear model

lm = sm.OLS(y_train,X_train_rfe).fit()   
#Let's see the summary of our linear model

print(lm.summary())
# Checking VIF

def check_vif(df):
    vif = pd.DataFrame()
    X = df
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    print(vif)

check_vif(X_train_rfe)
# Dropping a feature, Modelling, Checking VIF

X_train_new = X_train_rfe.drop(["FuelEconomy"], axis = 1)

lm = sm.OLS(y_train,X_train_new).fit()  

print(lm.summary()) 

check_vif(X_train_new)
# Dropping a feature, Modelling, Checking VIF

X_train_new = X_train_new.drop(["hardtop"], axis = 1)

lm = sm.OLS(y_train,X_train_new).fit()   

print(lm.summary()) 

check_vif(X_train_new)
# Dropping a feature, Modelling, Checking VIF

X_train_new = X_train_new.drop(["sedan"], axis = 1)

lm = sm.OLS(y_train,X_train_new).fit()   

print(lm.summary())

check_vif(X_train_new)
# Dropping a feature, Modelling, Checking VIF

X_train_new = X_train_new.drop(["wheelbase"], axis = 1)

lm = sm.OLS(y_train,X_train_new).fit()  

print(lm.summary()) 

check_vif(X_train_new)
# Dropping a feature, Modelling, Checking VIF

X_train_new = X_train_new.drop(["hatchback"], axis = 1)

lm = sm.OLS(y_train,X_train_new).fit()   

print(lm.summary()) 

check_vif(X_train_new)
# Dropping a feature, Modelling, Checking VIF

X_train_new = X_train_new.drop(["horsepower"], axis = 1)

lm = sm.OLS(y_train,X_train_new).fit() 

print(lm.summary())

check_vif(X_train_new)

y_train_pred = lm.predict(X_train_new)
# Plot the histogram of the error terms
fig = plt.figure()
res = y_train - y_train_pred
sns.distplot(res, bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)    
# scaling the numeric values

df_test[numeric_columns] = scaler.transform(df_test[numeric_columns])
df_test
# splitting data

y_test = df_test.pop('price')
X_test = df_test
# understanding data

print(y_test.shape)

print(X_test.shape)
# dropping constant
X_train_new = X_train_new.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Exploring data

X_test_new
# Making predictions

y_pred = lm.predict(X_test_new)
y_pred
# evaluatingm model using r2_score

from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)
# Plotting y_test and y_pred to understand the pattern

fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)  
plt.ylabel('y_pred', fontsize=16) 
plt.xlabel('y_test', fontsize=18)                           
plt.show()
# stats info of the model

lm.summary()
