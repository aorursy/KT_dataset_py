# importing python packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from scipy import stats

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
# loading Ford dataset



df = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/ford.csv')

df.head()
# getting the shape of data



df.info
# checking column types



df.dtypes
# detecting missing values in each column



missing_values = df.isnull()



for column in missing_values.columns.values.tolist():

    print(column)

    print(missing_values[column].value_counts())

    print("")
# searching for duplicated rows



duplicated_rows = df[df.duplicated()]

duplicated_rows
df.drop_duplicates(inplace = True)
# models



df_mod = df['model'].value_counts().to_frame()

df_mod
# years



df['year'].value_counts().to_frame()
# dropping incorrect value



df.drop(df.loc[df['year'] == 2060].index, inplace = True)
# transmission



df['transmission'].value_counts().to_frame()
# fuel type



df['fuelType'].value_counts().to_frame()
# engine size



df['engineSize'].value_counts().to_frame()
# dropping incorrect value



df.drop(df.loc[df['engineSize'] == 0.0].index, inplace = True)
# prices grouped by models - bar chart



df_grp = df[['model', 'price']].groupby(by = ['model']).sum().sort_values(by = ['price'], ascending = False) 



df_grp.plot(kind = 'bar', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.xlabel('Models', fontsize = 14)

plt.ylabel('Price', fontsize = 14)



plt.show()
# prices grouped by transmission - box plot



plt.figure(figsize = (12, 8))

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.xlabel('Transmission type', fontsize = 14)

plt.ylabel('Price', fontsize = 14)



sns.boxplot(x = 'transmission', y = 'price', data = df, palette = 'Greens')
# number of sales by models - horizontal bar chart



df_mod.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.xlabel('Used Car Sales ', fontsize = 14)

plt.ylabel('Models', fontsize = 14)



plt.show()
# prices grouped by fuel types - box plot



plt.figure(figsize = (12, 8))

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.xlabel('Fuel Type', fontsize = 14)

plt.ylabel('Price', fontsize = 14)



sns.boxplot(x = 'fuelType', y = 'price', data = df, palette = 'Blues')
# getting statistical summary of numeric-typed columns



df.describe()
# checking correlations



df.corr()
# getting graphical overview with pair plots



sns.pairplot(df)
# year and price - regression plot



sns.regplot(x = 'year', y = 'price', data = df, color = 'mediumaquamarine')

plt.ylim(0,)
# mileage and price - regression plot



sns.regplot(x = 'mileage', y = 'price', data = df, color = 'palevioletred')

plt.ylim(0,)
# mileage and year - regression plot



sns.regplot(x = 'mileage', y = 'year', data = df, color = 'steelblue')
# tax and price - regression plot



sns.regplot(x = 'tax', y = 'price', data = df, color = 'lightgrey')
# engine size and price - regression plot



sns.regplot(x = 'engineSize', y = 'price', data = df, color = 'plum')
# miles per gallon and price - regression plot



sns.regplot(x = 'mpg', y = 'price', data = df, color = 'tan')

plt.ylim(0,)
# year and engine size



df['year'] = df['year'] / df['year'].max()

df['mileage'] = df['mileage'] / df['mileage'].max()

df['engineSize'] = df['engineSize'] / df['engineSize'].max()



df[['year', 'mileage', 'engineSize']].head()
df['fuelType'] = pd.get_dummies(df['fuelType'])

df['fuelType'] = df['fuelType'].astype('int')
# fitting and predicting



X = df[['year']]

Y = df[['price']]



lm_year = LinearRegression()



lm_year.fit(X, Y)



Yhat = lm_year.predict(X)

Yhat[0:5]
# visualizing residuals



sns.residplot(df['year'], df['price'])

plt.show()
# determining model accuracy with R^2 and MSE



print("The R-square of train data is: ", lm_year.score(X, Y))

print("The mean squared error is: ", mean_squared_error(df['price'], Yhat))
# fitting and predicting



lm_ma = LinearRegression()



X = df[['mileage']]

Y = df[['price']]



lm_ma.fit(X, Y)



Yhat = lm_ma.predict(X)

Yhat[0:5]
# visualizing residuals



sns.residplot(df['mileage'], df['price'])

plt.show()
# determining model accuracy with R^2 and MSE



print("The R-square is: ", lm_ma.score(X, Y))

print("The mean squared error is: ", mean_squared_error(df['price'], Yhat))
# fitting and predicting



lm_es = LinearRegression()



X = df[['engineSize']]

Y = df[['price']]



lm_es.fit(X, Y)



Yhat = lm_es.predict(X)

Yhat[0:5]
# visualizing residuals



sns.residplot(df['engineSize'], df['price'])

plt.show()
# determining model accuracy with R^2 and MSE



print("The R-square is: ", lm_es.score(X, Y))

print("The mean squared error is: ", mean_squared_error(df['price'], Yhat))
# fitting and predicting



lm_ft = LinearRegression()



X = df[['fuelType']]

Y = df[['price']]



lm_ft.fit(X, Y)



Yhat = lm_ft.predict(X)

Yhat[0:5]
# visualizing residuals



sns.residplot(df['fuelType'], df['price'])

plt.show()
# determining model accuracy with R^2 and MSE



print("The R-square is: ", lm_ft.score(X, Y))

print("The mean squared error is: ", mean_squared_error(df['price'], Yhat))
# fitting and predicting



lm_multi = LinearRegression()



X = df[['year', 'mileage', 'engineSize', 'fuelType']]

Y = df[['price']]



lm_multi.fit(X, Y)



Yhat = lm_multi.predict(X)

Yhat[0:5]
# determining model accuracy with R^2 and MSE



print("The R-square is: ", lm_multi.score(X, Y))

print("The mean squared error is: ", mean_squared_error(df['price'], Yhat))
# Processing data using Pipeline



X = df[['year', 'mileage', 'engineSize', 'fuelType']]

Y = df['price']



pipeline = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree = 2)), ('mode', LinearRegression())]

poly_model = Pipeline(pipeline)
# fitting and predicting



X = df[['year', 'mileage', 'engineSize', 'fuelType']]

Y = df[['price']]



poly_model.fit(X, Y)



Yhat = poly_model.predict(X)

Yhat[0:5]
# determining model accuracy with R^2 and MSE



print("The R-square is: ", poly_model.score(X, Y))

print("The mean squared error is: ", mean_squared_error(df['price'], Yhat))
# Processing data using Pipeline



X = df[['year', 'mileage', 'engineSize', 'fuelType']]

Y = df['price']



pipeline = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree = 13)), ('mode', LinearRegression())]

poly_model = Pipeline(pipeline)
# fitting and predicting



X = df[['year', 'mileage', 'engineSize', 'fuelType']]

Y = df[['price']]



poly_model.fit(X, Y)



Yhat = poly_model.predict(X)

Yhat[0:5]
# determining model accuracy with R^2 and MSE



print("The R-square is: ", poly_model.score(X, Y))

print("The mean squared error is: ", mean_squared_error(df['price'], Yhat))