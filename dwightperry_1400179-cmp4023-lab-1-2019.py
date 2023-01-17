# -*- coding: utf-8 -*-

"""

Created on Tue Mar 11 22:55:01 2019

Data Warehouse & Mining Lab 1 Assignment

@authors: Dwayne Perry

ID# : 1400179

"""





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import os



import random
df = pd.DataFrame(np.random.randint(1.0,500.0,size=(50,3)),

              index = range(0,50),

              columns = ['X1','X2','Y'],

              dtype = 'float64')
print("Data frame shape : ", df.shape)



df
# Correlation between X1 and Y.

corr_1 = df['X1'].corr(df['Y'])



print("Correlation between X1 and Y : ", corr_1)
# Correlation between X2 and Y.

corr_1 = df['X2'].corr(df['Y'])



print("Correlation between X1 and Y : ", corr_1)
df.corr()
print("Correlation between DataFrame :") 

f, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
plt.scatter(X1, Y)

plt.title('Correlation between X2 and Y')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
plt.scatter(X2, Y)

plt.title('Correlation between X2 and Y')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
sns.pairplot(df)
X = df[['X1', 'X2']]

y = df['Y']
# Train Test Split.



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
# Creating and training the model.



from sklearn.linear_model import LinearRegression
lm = LinearRegression()



lm.fit(X_train,y_train)
# Model evaluation.



# print the intercept

print("Intercept value : ", lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
# Predictions from the model.



y_predictions = lm.predict(X_test)
plt.scatter(y_test, y_predictions)

plt.title('Residual Plot')

plt.xlabel('y_test')

plt.ylabel('predictions')

plt.show()
sns.distplot((y_test-y_predictions),bins=50);

plt.title('Residual Histogram')

from sklearn.metrics import r2_score



# The coefficient of Determination (R^2) of your model.

coefficient_of_dermination = r2_score(y_test, y_predictions)



print("Coefficient of Determination (R^2) : ", coefficient_of_dermination)



# The coefficient of determination is the square of the correlation (r) between predicted y scores and actual y scores;

# thus, it ranges from 0 to 1. However, when the (R^2) value is negative such as in this case;

# because the chosen model fits worse than a horizontal line, then R2 is negative.

# Perhaps because the model was created b randomly genearated data.

lm.score(X,y)



# Regression formula.

# Formula Y = X1 + X2 + E

# Y = -0.032859(X1) + 0.117294(X2) + 235.05178462615243
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_predictions))

print('MSE:', metrics.mean_squared_error(y_test, y_predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predictions)))





# a. MAE is the easiest to understand, because it's the average error.



# b. RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units. All of these are loss functions, 

# because we want to minimize them.



# c. MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.