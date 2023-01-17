# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
X = np.array([[-40], [-10],  [0],  [8], [15], [22]],  dtype=float)

y= np.array([[-40],  [14], [32], [46], [59], [72]],  dtype=float)
import matplotlib.pyplot as plt

plt.xlabel('X')

plt.ylabel("y")

plt.scatter(X,y)
reg = LinearRegression().fit(X, y)

reg.score(X, y)
reg.coef_
reg.intercept_ 
reg.predict(np.array([[38]]))