#importing libraries 

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt
# Importing Data

data = pd.read_csv( '../input/dp-lr-model-train-cleaned-datset/train_cleaned.csv')

data.head()
#seperating independent and dependent variables

x = data.drop(['Item_Outlet_Sales'], axis=1)

y = data['Item_Outlet_Sales']

x.shape, y.shape
# Importing the train test split function

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56)
#importing Linear Regression and metric mean square error

from sklearn.linear_model import LinearRegression as LR

from sklearn.metrics import mean_absolute_error as mae
# Creating instance of Linear Regresssion

lr = LR()



# Fitting the model

lr.fit(train_x, train_y)
# Predicting over the Train Set and calculating error

train_predict = lr.predict(train_x)

k = mae(train_predict, train_y)

print('Training Mean Absolute Error', k )
# Predicting over the Test Set and calculating error

test_predict = lr.predict(test_x)

k = mae(test_predict, test_y)

print('Test Mean Absolute Error    ', k )
lr.coef_
plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='b')

x = range(len(train_x.columns))

y = lr.coef_

plt.bar( x, y )

plt.xlabel( "Variables")

plt.ylabel('Coefficients')

plt.title('Coefficient plot')
# Arranging and calculating the Residuals

residuals = pd.DataFrame({

    'fitted values' : test_y,

    'predicted values' : test_predict,

})



residuals['residuals'] = residuals['fitted values'] - residuals['predicted values']

residuals.head()
plt.figure(figsize=(10, 6), dpi=120, facecolor='w', edgecolor='b')

f = range(0,2131)

k = [0 for i in range(0,2131)]

plt.scatter( f, residuals.residuals[:], label = 'residuals')

plt.plot( f, k , color = 'red', label = 'regression line' )

plt.xlabel('fitted points ')

plt.ylabel('residuals')

plt.title('Residual plot')

plt.ylim(-4000, 4000)

plt.legend()
# Histogram for distribution

plt.figure(figsize=(10, 6), dpi=120, facecolor='w', edgecolor='b')

plt.hist(residuals.residuals, bins = 150)

plt.xlabel('Error')

plt.ylabel('Frequency')

plt.title('Distribution of Error Terms')

plt.show()
# importing the QQ-plot from the from the statsmodels

from statsmodels.graphics.gofplots import qqplot



## Plotting the QQ plot

fig, ax = plt.subplots(figsize=(5,5) , dpi = 120)

qqplot(residuals.residuals, line = 's' , ax = ax)

plt.ylabel('Residual Quantiles')

plt.xlabel('Ideal Scaled Quantiles')

plt.title('Checking distribution of Residual Errors')

plt.show()
# Importing Variance_inflation_Factor funtion from the Statsmodels

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant



# Calculating VIF for every column (only works for the not Catagorical)

VIF = pd.Series([variance_inflation_factor(data.values, i) for i in range(data.shape[1])], index =data.columns)

VIF
# Creating instance of Linear Regresssion

lr = LR(normalize = True)



# Fitting the model

lr.fit(train_x, train_y)
# Predicting over the Train Set and calculating error

train_predict = lr.predict(train_x)

k = mae(train_predict, train_y)

print('Training Mean Absolute Error', k )
# Predicting over the Test Set and calculating error

test_predict = lr.predict(test_x)

k = mae(test_predict, test_y)

print('Test Mean Absolute Error    ', k )
plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='b')

x = range(len(train_x.columns))

y = lr.coef_

plt.bar( x, y )

plt.xlabel( "Variables")

plt.ylabel('Coefficients')

plt.title('Normalized Coefficient plot')
#seperating independent and dependent variables

x = data.drop(['Item_Outlet_Sales'], axis=1)

y = data['Item_Outlet_Sales']

x.shape, y.shape
Coefficients = pd.DataFrame({

    'Variable'    : x.columns,

    'coefficient' : lr.coef_

})

Coefficients.head()
sig_var = Coefficients[Coefficients.coefficient > 0.5]
subset = data[sig_var['Variable'].values]

subset.head()
# Importing the train test split function

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(subset, y , random_state = 56)
#importing Linear Regression and metric mean square error

from sklearn.linear_model import LinearRegression as LR

from sklearn.metrics import mean_absolute_error as mae
# Creating instance of Linear Regresssion with Normalised Data

lr = LR(normalize = True)



# Fitting the model

lr.fit(train_x, train_y)
# Predicting over the Train Set and calculating error

train_predict = lr.predict(train_x)

k = mae(train_predict, train_y)

print('Training Mean Absolute Error', k )
# Predicting over the Test Set and calculating error

test_predict = lr.predict(test_x)

k = mae(test_predict, test_y)

print('Test Mean Absolute Error    ', k )
plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='b')

x = range(len(train_x.columns))

y = lr.coef_

plt.bar( x, y )

plt.xlabel( "Variables")

plt.ylabel('Coefficients')

plt.title('Normalized Coefficient plot')