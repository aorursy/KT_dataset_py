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
dataframe = pd.read_csv('/kaggle/input/weight-height/weight-height.csv')
dataframe.head()
heights = np.array(dataframe['Height'])

weights = np.array(dataframe['Weight']).reshape(-1, 1)

heights.shape
weights.shape
import matplotlib.pyplot as plt
plt.plot(weights, heights, 'bs')
from sklearn.linear_model import LinearRegression
my_linear_model = LinearRegression()
my_linear_model.fit(weights, heights)
heights_prediction = my_linear_model.predict(weights)
plt.plot(weights, heights, 'bs')

plt.plot(weights, heights_prediction, 'ro')
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(heights, heights_prediction)
mse
mse**0.5