"""

Created on Sun Mar 15 20:16:55 2020



@author: shoaib.zafer

"""

import matplotlib.pyplot as plt

import numpy as np

from sklearn import linear_model

X = 2 * np.random.rand(100,1)

y = 4 +3 * X+np.random.randn(100,1)

plt.scatter(X,y)

lm = linear_model.LinearRegression()

model = lm.fit(X,y)

predictions = lm.predict(X)

plt.plot(X,predictions,"g")

plt.show()

print("R² score=" , lm.score(X,y)) 

print("solpe =",lm.coef_ )

print("intercept=", lm.intercept_ )