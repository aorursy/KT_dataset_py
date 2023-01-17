import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/simple-linear-regression/kc_house_data.csv')
data.head()
x = data['sqft_living'].values.reshape(-1,1)
y = data['price'].values.reshape(-1,1)
plt.figure(figsize=(16, 8))
plt.scatter(
    data['sqft_living'],
    data['price'],
    c='blue'
)
plt.xlabel("Living Area")
plt.ylabel("Price")
plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
reg = LinearRegression()
reg.fit(x_train, y_train)
print("The linear model is: Price = {:.5} + {:.5}(Living Area)".format(reg.intercept_[0], reg.coef_[0][0]))
predictions = reg.predict(x_train)
plt.figure(figsize=(16, 8))
plt.scatter(
     x_train,
     y_train,
     c='blue'
)
plt.plot(
    x_train,
    predictions,
    c='black',
    linewidth=2
)
plt.xlabel("Living Area")
plt.ylabel("Price")
plt.show()
predictions = reg.predict(x_test)
plt.figure(figsize=(16, 8))
plt.scatter(
     x_test,
     y_test,
     c='blue'
)
plt.plot(
    x_test,
    predictions,
    c='black',
    linewidth=2
)
plt.xlabel("Experience")
plt.ylabel("Stipend")
plt.show()
print(mean_absolute_error(y_test, predictions))
print(mean_squared_error(y_test, predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))