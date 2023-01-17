import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import r2_score

%matplotlib inline
data = pd.read_csv('/kaggle/input/oc2emission/FuelConsumptionCo2.csv')

data.columns
sns.swarmplot(x="FUELTYPE", y="CO2EMISSIONS", data=data);
sns.swarmplot(x="CYLINDERS", y="CO2EMISSIONS", data=data);
redundant_cols = ['MODELYEAR','MAKE','MODEL','VEHICLECLASS','TRANSMISSION','FUELTYPE']

data.drop(redundant_cols, axis=1, inplace=True)

# data.sample(10)
sns.pairplot(data);
data.corr()
sns.heatmap(data.corr());
data['INV_FUELCONSUMPTION_COMB_MPG'] = 1/data['FUELCONSUMPTION_COMB_MPG']
plt.scatter(data['INV_FUELCONSUMPTION_COMB_MPG'], data['CO2EMISSIONS']);
plt.scatter(data['FUELCONSUMPTION_COMB'], data['CO2EMISSIONS']);
data.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','INV_FUELCONSUMPTION_COMB_MPG','FUELCONSUMPTION_COMB'],axis=1, inplace=True)
data.columns
X = data.drop('CO2EMISSIONS',axis=1)

y = data['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
def measure(actual, prediction):

    print('Mean Absolute Error:', metrics.mean_absolute_error(actual, prediction))  

    print('Mean Squared Error:', metrics.mean_squared_error(actual, prediction))  

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(actual, prediction)))

    print('r2 score:',r2_score(prediction, actual))

    

measure(y_test,y_test)
regressor = LinearRegression()  

regressor.fit(X_train.loc[:,['ENGINESIZE']], y_train)

measure(y_test, regressor.predict(X_test.loc[:,['ENGINESIZE']]))
cols = ['ENGINESIZE','FUELCONSUMPTION_COMB_MPG']

regressor = LinearRegression()  

regressor.fit(X_train.loc[:,cols], y_train)

yhat = regressor.predict(X_test.loc[:,cols])

measure(y_test, yhat)

print(regressor.coef_)
sns.residplot(y_test, yhat)

plt.title('Residual plot of YHAT x Y_TEST')

plt.show()
sns.distplot(y_test, hist = False, label = 'Actual values')

sns.distplot(yhat, hist = False, label = 'Predicted values')

plt.title('Comparison of predicted values with actual values')

plt.show()
df = data.copy()
from sklearn import preprocessing



x = df.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df = pd.DataFrame(x_scaled)

df.columns = ['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']
X = df.drop('CO2EMISSIONS',axis=1)

y = df['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
cols = ['ENGINESIZE','FUELCONSUMPTION_COMB_MPG']

regressor = LinearRegression()  

regressor.fit(X_train.loc[:,cols], y_train)

yhat = regressor.predict(X_test.loc[:,cols])

measure(y_test, yhat)

print(regressor.coef_)
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model

train_x = np.asanyarray(X_train[['ENGINESIZE']])

train_y = np.asanyarray(y_train)



test_x = np.asanyarray(X_test[['ENGINESIZE']])

test_y = np.asanyarray(y_test)





poly = PolynomialFeatures(degree=2)

train_x_poly = poly.fit_transform(train_x)

test_x_poly = poly.transform(test_x)



regressor = LinearRegression()  

regressor.fit(train_x_poly, train_y)

yhat = regressor.predict(test_x_poly)

measure(y_test, yhat)

print(regressor.coef_)