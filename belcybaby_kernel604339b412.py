# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

data=data.dropna()

X=data['x'].values

Y=data['y'].values

m=len(Y)

X = X.reshape((m, 1))

# Creating Model

reg = LinearRegression()

# Fitting training data

reg = reg.fit(X, Y)

# Y Prediction

Y_pred = reg.predict(X)



# Calculating RMSE and R2 Score

mse = mean_squared_error(Y, Y_pred)

rmse = np.sqrt(mse)

r2_score = reg.score(X, Y)



print(np.sqrt(mse))

print(r2_score)