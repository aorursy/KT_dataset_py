#Importing the pandas for data processing and numpy for numerical computing

import numpy as np

import pandas as pd
# Importing the Boston Housing dataset from the sklearn

from sklearn.datasets import load_boston

boston = load_boston()
#Converting the data into pandas dataframe

data = pd.DataFrame(boston.data)
#First look at the data

data.head()
#Adding the feature names to the dataframe

data.columns = boston.feature_names
#Adding the target variable to the dataset

data['PRICE'] = boston.target 
#Looking at the data with names and target variable

data.head()
#Shape of the data

print(data.shape)
#Checking the null values in the dataset

data.isnull().sum()
#Checking the statistics of the data

data.describe()
data.info()
#checking the distribution of the target variable

import seaborn as sns

sns.distplot(data.PRICE)
#Distribution using box plot

sns.boxplot(data.PRICE)
#checking Correlation of the data 

correlation = data.corr()

correlation.loc['PRICE']
# plotting the heatmap

import matplotlib.pyplot as plt

fig,axes = plt.subplots(figsize=(15,12))

sns.heatmap(correlation,square = True,annot = True)
# Checking the scatter plot with the most correlated features

plt.figure(figsize = (20,5))

features = ['LSTAT','RM','PTRATIO']

for i, col in enumerate(features):

    plt.subplot(1, len(features) , i+1)

    x = data[col]

    y = data.PRICE

    plt.scatter(x, y, marker='o')

    plt.title("Variation in House prices")

    plt.xlabel(col)

    plt.ylabel('"House prices in $1000"')
#X = data[['LSTAT','RM','PTRATIO']]

X = data.iloc[:,:-1]

y= data.PRICE
# Splitting the data into train and test for building the model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)
#Linear Regression 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
#Fitting the model

regressor.fit(X_train,y_train)
#Prediction on the test dataset

y_pred = regressor.predict(X_test)
# Predicting RMSE the Test set results

from sklearn.metrics import mean_squared_error

rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

print(rmse)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print(r2)
#Scaling the dataset

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Creating the neural network model

import keras

from keras.layers import Dense, Activation,Dropout

from keras.models import Sequential



model = Sequential()



model.add(Dense(128,activation  = 'relu',input_dim =13))

model.add(Dense(64,activation  = 'relu'))

model.add(Dense(32,activation  = 'relu'))

model.add(Dense(16,activation  = 'relu'))

model.add(Dense(1))

model.compile(optimizer = 'adam',loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 100)
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print(r2)
# Predicting RMSE the Test set results

from sklearn.metrics import mean_squared_error

rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

print(rmse)