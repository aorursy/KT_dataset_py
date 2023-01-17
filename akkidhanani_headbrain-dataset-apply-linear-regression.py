import pandas as pd

%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

headbrain = pd.read_csv("../input/headbrain.csv")
headbrain.head(10)
plt.rcParams['figure.figsize']=(20.0, 10.0)
x=headbrain['Head Size(cm^3)'].values

y=headbrain['Brain Weight(grams)'].values
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score
lm=linear_model.LinearRegression()
x_input = x.reshape(237,1)

y_input = y.reshape(237,1)
lm.fit(x_input, y_input)
y_pred = lm.predict(x_input)
print("Mean squared error: %.2f"% np.sqrt(mean_squared_error(y_input, y_pred)))
plt.scatter(x_input, y_input)

plt.plot(x,y_pred)