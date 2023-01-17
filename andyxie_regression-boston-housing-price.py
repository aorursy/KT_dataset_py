import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from sklearn import datasets



boston = datasets.load_boston()
type(boston)
X = boston["data"]

y = boston["target"]
X.shape, y.shape
col_names = boston["feature_names"]

col_names
plt.scatter(X[:,5], y, alpha=0.4)

plt.show()