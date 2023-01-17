import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
df = pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv').dropna()

df.head()
x = df[['wheelbase', 'carlength', 'carwidth', 'carheight', 'enginesize', 'boreratio', 'horsepower']]

y = df[['price']]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
# Model

regressor = LinearRegression()

regressor.fit(X_train, Y_train)
regressor.score(X_test, Y_test)