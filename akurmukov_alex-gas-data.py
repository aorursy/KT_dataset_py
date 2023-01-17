# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/Alex_gas_data - Sheet1.csv')
import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
data.head()
fig, ax = plt.subplots(1,3, figsize=(25,5))

results = {}

for i,gas in enumerate(['air', 'nitrogen', 'oxigen']):

    X = data.dropna(subset=[f'X_{gas}'])[f'X_{gas}'].values.reshape(-1,1)

    y = data.dropna(subset=[f'X_{gas}'])[f'Y_{gas}'].values.reshape(-1,1)

    

    # generate features

    poly = PolynomialFeatures(degree=3)

    _X = poly.fit_transform(X)

    

    # generate plotting grid

    x_grid = np.arange(X.min(), X.max(), 0.25).reshape(-1,1)

    _x_grid = poly.transform(x_grid) 

    

    # create model

    reg = LinearRegression(fit_intercept=False, normalize=False)

    reg.fit(_X, y)

    

    # add formula

    temp = []

    for j,val in enumerate(reg.coef_.flatten()):

        temp.append(f'x^{j} * {np.round(val, 1)}')

    text = ' + '.join(temp)

    

    # plot results

    y_pred = reg.predict(_x_grid)

    ax[i].scatter(X, y)

    ax[i].plot(x_grid, y_pred, label=text)

    ax[i].set_title(gas)

    ax[i].legend()

     

    results[f'X_{gas}'] = list(np.round(x_grid.flatten(), 3))

    results[f'y_{gas}'] = list(np.round(y_pred.flatten()))
import json
results
{'X_air': [0.34, 0.59, 0.84, 1.09, 1.34, 1.59, 1.84],

 'y_air': [2000.0, 2165.0, 2203.0, 2160.0, 2080.0, 2007.0, 1985.0],

 'X_nitrogen': [0.29, 0.54, 0.79, 1.04, 1.29, 1.54, 1.79, 2.04],

 'y_nitrogen': [2670.0,

  2768.0,

  2739.0,

  2626.0,

  2475.0,

  2328.0,

  2230.0,

  2224.0],

 'X_oxigen': [0.33, 0.58, 0.83, 1.08, 1.33, 1.58, 1.83, 2.08, 2.33, 2.58],

 'y_oxigen': [1110.0,

  1179.0,

  1213.0,

  1218.0,

  1204.0,

  1177.0,

  1144.0,

  1114.0,

  1094.0,

  1092.0]}