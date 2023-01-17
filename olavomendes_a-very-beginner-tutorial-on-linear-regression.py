# Importing the necessary libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
# Creating a very simple dictionary containing the "age" and "IQ" of some people

data = {'Age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],

         'QI': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]

        }
# Turning into a DataFrame

df = pd.DataFrame(data)

df
# Seeing the correlation of features.

# Correlation = 1 means they are fully related

# Correlation = -1 means that the are little related

# Correlation = 0 means that there is no relationship

df.corr()
# Separating the independent variable (X) and the dependent variable (y)

# "Age" is the variable with the values we use as a basis for the forecast

# The "IQ" is the variable with the values that we want to predict

X = df.drop('QI', axis=1)

y = df.Age



# Defining our training model

model = LinearRegression()



# Separating the model into training data and test data. I set 80% of the training dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)



model.fit(X_train, y_train)
# Our equation of the line is given by y = mx + b



b = model.coef_[0] #slope

m = model.intercept_ #linear coefficient



# Showing the equation of our line

print(f'Line equation: y = {b:.1f}x + {m:.1f}')
# Making predictions

preds = model.predict(X_test)

preds
# Creating a DataFrame to show the predicted values ​​and compare with the real ones. We got 100% correct.

compare = pd.DataFrame(X_test)

compare['Preds'] = preds

compare['Real'] = y_test

compare
# A graph illustrating the linear function that the algorithm defined in training

sns.lmplot(x='Age', y='Preds', data=compare);