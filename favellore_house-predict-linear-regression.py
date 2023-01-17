# This model is using Linear Regression.
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

%matplotlib inline
# Importing the dataset and extracting the Independent and Dependent Variables



house = pd.read_csv('../input/house.csv')

X = house.iloc[:,:-1].values

y = house.iloc[:,20].values



house.head()
y
#Data Visualization

# Corelation of various parameters



fig, ax = plt.subplots(figsize=(20,15))



sns.heatmap(house.corr())
# Encoding Categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()



X[:,1] = labelencoder.fit_transform(X[:,1])



onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()
# Avoiding dummy variables



X = X[:,1:]
# Splitting the dataset into the Training set and Test set



from sklearn.model_selection import train_test_split

X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
# Fitting Multiple Linear Regression to the Training set



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
# Predicting the Test set results

y_pred = regressor.predict(X_test)

print(y_pred)
# Calculating the Co-efficient

print(regressor.coef_)
# Calculating the Intercept

print(regressor.intercept_)
# Calculating the R Squared Value

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
fig=sns.jointplot(y='price',x='bedrooms',data=house)

plt.show()
ax = sns.boxplot(x=house['price'])
ay = sns.boxplot(x='bedrooms',y='price', data = house, width= 0.6)
house.groupby('bedrooms')['price'].describe()