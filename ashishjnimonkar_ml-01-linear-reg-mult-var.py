# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.datasets import make_regression

from matplotlib import pyplot



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#X, y = make_regression(n_samples=100, n_features=1, noise=5)

X, y = make_regression(n_samples=100, n_features=4, noise=5)



#pyplot.scatter(X,y)

#pyplot.show()



print(X)

print(y)