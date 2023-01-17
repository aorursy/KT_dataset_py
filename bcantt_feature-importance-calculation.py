# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/car-data/CarPrice_Assignment.csv')
data
le = preprocessing.LabelEncoder()

for name in data.columns:

    if data[name].dtypes == "O":

        print(name)

        data[name] = data[name].astype(str)

        le.fit(data[name])

        data[name] = le.transform(data[name])

var_changes = {}

for name in data.columns:

    var_changes[name] = 0
var_changes = {}

for name in data.columns:

    the_list = []

    for i in range(30):

        the_list.append((data.iloc[int(i*(data.shape[0]/30)):int((i+1)*(data.shape[0]/30))]['price'].var()+0.01) / (data.iloc[int(i*(data.shape[0]/30)):int((i+1)*(data.shape[0]/30))][name].var()+0.01) / len(data[name].unique()))

        

    var_changes[name] = [float(i)/sum(the_list) for i in the_list]
data.corr().sort_values('price',ascending = False)
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')

plt.plot(var_changes['enginesize'],label = 'enginesize')

plt.plot(var_changes['curbweight'],label = 'curbweight')

plt.plot(var_changes['horsepower'],label = 'horsepower')

plt.plot(var_changes['carlength'],label = 'carlength')

plt.plot(var_changes['drivewheel'],label = 'drivewheel')

plt.plot(var_changes['wheelbase'],label = 'wheelbase')

plt.plot(var_changes['boreratio'],label = 'boreratio')

plt.plot(var_changes['fuelsystem'],label = 'fuelsystem')

plt.plot(var_changes['citympg'],label = 'citympg')

plt.plot(var_changes['highwaympg'],label = 'highwaympg')

plt.legend(loc="upper right")



plt.show()