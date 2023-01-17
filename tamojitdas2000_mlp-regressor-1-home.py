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
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv("../input/data.csv")

plt.plot(data['x'],data['y'])

plt.show()
from sklearn.neural_network import MLPRegressor



y=data.pop('y')

x=data



model=MLPRegressor(hidden_layer_sizes=[50,50,50,50],max_iter=40000)

model.fit(x,y)

y_predicted=model.predict(x)



plt.plot(x,y,color='red')

plt.plot(x,y_predicted,color='green')

plt.show()
model=MLPRegressor(hidden_layer_sizes=[500,500,500,500],max_iter=60000)

model.fit(x,y)

y_predicted=model.predict(x)



plt.plot(x,y,color='red')

plt.plot(x,y_predicted,color='green')

plt.show()
model=MLPRegressor(hidden_layer_sizes=[50,50,50,50],max_iter=40000,activation='tanh')

model.fit(x,y)

y_predicted=model.predict(x)



plt.plot(x,y,color='red')

plt.plot(x,y_predicted,color='green')

plt.show()
model=MLPRegressor(hidden_layer_sizes=[500,800,800,500],max_iter=60000,activation='tanh')

model.fit(x,y)

y_predicted=model.predict(x)



plt.plot(x,y,color='red')

plt.plot(x,y_predicted,color='green')

plt.show()