import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

print('Libraries imported successfully')
df = pd.read_csv('../input/FuelConsumptionCo2.csv')

print('Dataset imported successfully')
df.shape
df.head()
df.info()
df.corr()
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

print('Dataframe CDF created successfully')
cdf.head()
cdf.corr()
sns.regplot(cdf.ENGINESIZE, cdf.CO2EMISSIONS)

sns.regplot(cdf.CYLINDERS, cdf.CO2EMISSIONS)

sns.regplot(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS)

plt.ylabel('CO2 Emission')

plt.legend(('Engine Size', 'Cylinders', 'Fuel Consupmtion'))

plt.show()
# creating x,y

x = cdf.iloc[:,0:3]

y = cdf.CO2EMISSIONS

y = y.to_frame()
# taking a look at x,y
x.head()
y.head()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
print ('x_train quantity: ', len(x_train))

print ('y_train quantity: ', len(y_train))

print ('x_test quantity: ', len(x_test))

print ('y_test quantity: ', len(y_test))
lm = LinearRegression()
lm.fit(x_train, y_train)
# b0 = lm.intercept_ 

# b1 = lm.coef_
lm.intercept_
lm.coef_
yhat = lm.predict(x_test)
lm.score(x_test, y_test)
yhatdf = pd.DataFrame(yhat)

yhatdf.columns = ['PredictedValues']

yhatdf = yhatdf.PredictedValues.astype(int).to_frame()
actualval = y_test

actualval.columns = ['ActualValues']
lastdf = pd.concat([yhatdf.reset_index(drop=True), actualval.reset_index(drop=True)], axis = 1, sort = False)
lastdf
sns.residplot(y_test, yhat)

plt.title('Residual plot of YHAT x Y_TEST')

plt.show()
sns.distplot(y_test, hist = False, label = 'Actual values')

sns.distplot(yhat, hist = False, label = 'Predicted values')

plt.title('Comparison of predicted values with actual values')

plt.show()