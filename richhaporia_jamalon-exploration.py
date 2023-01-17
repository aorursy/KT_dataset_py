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

import numpy as np

import matplotlib.pyplot as plt



import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures 



from sklearn import metrics

%matplotlib inline
dataset = pd.read_csv("../input/jamalon-arabic-books-dataset/jamalon dataset.csv")

dataset
x_name = 'Pages'

y_name = 'Price'

dataset = dataset.sort_values(by=[x_name])

dataset = dataset[dataset.Pages != 0]

dataset = dataset[dataset.Pages <= 5000]
plt.scatter(dataset.Pages, dataset.Price)
line = LinearRegression()  

line.fit(dataset[[x_name]], dataset[[y_name]]) 



fig = plt.figure()

ax = plt.gca()

plt.title('Price vs. Pages')

plt.xlabel('Pages (log)')

plt.ylabel('Price (log)')

ax.plot(dataset[x_name] ,dataset[y_name], 'o', c='blue', alpha=0.1, markeredgecolor='none')

ax.set_yscale('log')

ax.set_xscale('log')



plt.plot(dataset[[x_name]], line.predict(dataset[[x_name]]), color = 'red')



plt.show()
quantileCount = 4 #splitting data into quarters

quantiles = pd.qcut(dataset[y_name], 4, labels = False)

dataset['Quantile_Category'] = quantiles
fig = plt.figure()

ax = plt.gca()

plt.title('Price vs. Pages')

plt.xlabel('Pages (log)')

plt.ylabel('Price (log)')

ax.plot(dataset[x_name] ,dataset[y_name], 'o', c='blue', alpha=0.1, markeredgecolor='none')

ax.set_yscale('log')

ax.set_xscale('log')





for x in range(quantileCount):

    quantileData = dataset[dataset['Quantile_Category'] == x]

    quantileLine = LinearRegression()

    quantileLine.fit(quantileData[[x_name]], quantileData[[y_name]])

    plt.plot(dataset[[x_name]], quantileLine.predict(dataset[[x_name]]), color = 'red')

    

    

#plt.plot(dataset[['Pages']], line.predict(dataset[['Pages']]), color = 'red')

plt.show()