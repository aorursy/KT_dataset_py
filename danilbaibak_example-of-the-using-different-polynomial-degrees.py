# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# get x and y vectors
x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0,  2.0,  4.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0, 0.7, 0.9])

# calculate polynomial
z = np.polyfit(x, y, 3)
my_poly1d = np.poly1d(z, r=True)
print(my_poly1d)
# calculate new x's and y's
x_new = np.linspace(x[0], x[-1], 100).reshape(-1, 1)
y_new = my_poly1d(x_new)
plt.scatter(x_new, y_new, label='training points');
plt.legend();
X = x_new[:85].reshape(-1, 1)
y = y_new[:85]
for count, degree in enumerate([1, 2, 3]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.001))
    model.fit(X, y)
    y_pred = model.predict(x_new)
    plt.plot(x_new, y_pred, linewidth=2, label='degree %d' % degree)

plt.scatter(x_new, y_new, s=30, marker='o', label='training points');
plt.legend(loc='upper left')
plt.show()
for count, degree in enumerate([1, 2, 3]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=10))
    model.fit(X, y)
    y_pred = model.predict(x_new)
    plt.plot(x_new, y_pred, linewidth=2, label='degree %d' % degree)

plt.scatter(x_new, y_new, s=30, marker='o', label='training points');
plt.legend(loc='upper left')
plt.show()
df = pd.DataFrame({'1': [1, 2, 3, 4], '2': [1, 4, 9, 16], '3': [1, 8, 27, 64]})
df.corr()



