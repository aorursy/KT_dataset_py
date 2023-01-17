import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df.head()
df.isnull().sum()
print(df.columns.values)

df.describe()

df.describe(include=['O'])

corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,cmap = "coolwarm",annot=True, fmt = ".2f")
df.drop(['id' ,'date','yr_built','zipcode','long','condition'], axis=1, inplace=True)
df.head()
sns.barplot(x='bedrooms', y='price', data=df)
plt.show()
sns.barplot(x='floors', y='price', data=df)
plt.show()
sns.barplot(x='grade', y='price', data=df)
plt.show()
plt.scatter(x='bathrooms', y='price', data=df)

plt.scatter(x='sqft_living', y='price', data=df)

df = df.drop(df[(df['sqft_living']>12000)& (df['price']>2)].index)
plt.scatter(x='sqft_living', y='price', data=df)

plt.scatter(x='sqft_lot', y='price', data=df)
plt.scatter(x='lat', y='price', data=df)


plt.scatter(x='sqft_above', y='price', data=df)

df = df.drop(df[(df['sqft_above']>7600)& (df['price']>1.1)].index)
plt.scatter(x='sqft_above', y='price', data=df)

df['price'] = df['price'] / df['price'].max()
df['bedrooms'] = df['bedrooms'] / df['bedrooms'].max()
df['bathrooms'] = df['bathrooms'] / df['bathrooms'].max()
df['sqft_living'] = df['sqft_living'] / df['sqft_living'].max()
df['sqft_lot'] = df['sqft_lot'] / df['sqft_lot'].max()
df['floors'] = df['floors'] / df['floors'].max()
df['waterfront'] = df['waterfront'] / df['waterfront'].max()
df['view'] = df['view'] / df['view'].max()
df['grade'] = df['grade'] / df['grade'].max()
df['sqft_above'] = df['sqft_above'] / df['sqft_above'].max()
df['sqft_basement'] = df['sqft_basement'] / df['sqft_basement'].max()
df['yr_renovated'] = df['yr_renovated'] / df['yr_renovated'].max()
df['lat'] = df['lat'] / df['lat'].max()
df['sqft_living15'] = df['sqft_living15'] / df['sqft_living15'].max()
df['sqft_lot15'] = df['sqft_lot15'] / df['sqft_lot15'].max()

df.head()


X = df.drop('price', axis=1)
y = df['price']

#Splitted Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Applying Gradient Boosting Regressor Model 
from sklearn.ensemble import GradientBoostingRegressor



GBRModel = GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate = 1.5 ,random_state=33)
GBRModel.fit(X_train, y_train)

print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))
print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = GBRModel.predict(X_test)
print('Predicted Value for GBRModel is : ' , y_pred[:10])

from sklearn.metrics import mean_absolute_error 

#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)
from sklearn.neighbors import KNeighborsRegressor




KNeighborsRegressorModel = KNeighborsRegressor(n_neighbors = 5, weights='uniform', #also can be : distance, or defined function 
                                               algorithm = 'auto')    #also can be : ball_tree ,  kd_tree  , brute
KNeighborsRegressorModel.fit(X_train, y_train)

#Calculating Details
print('KNeighborsRegressorModel Train Score is : ' , KNeighborsRegressorModel.score(X_train, y_train))
print('KNeighborsRegressorModel Test Score is : ' , KNeighborsRegressorModel.score(X_test, y_test))
#print('----------------------------------------------------')

#Calculating Prediction
y_pred = KNeighborsRegressorModel.predict(X_test)
print('Predicted Value for KNeighborsRegressorModel is : ' , y_pred[:10])
