# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for garph
import math # for math function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Lemonade.csv")
print(data.shape)
average = data['Sales'].mean()
print(average)
records_lower_than_average = data[data['Sales']<average]
records_lower_than_average.head()
#print(data['Sales'])
print(data.Temperature.max())
plt.scatter(data.Sales, data.Temperature,c='blue',marker = '+')
plt.title('scatter plot of sales and temperature')
plt.xlabel('Sales')
plt.ylabel('Temperatures')

p = data.groupby('Day')['Sales'].agg(['sum'])
p2 = data.groupby('Day')['Sales'].agg(['count'])
p3 = data.groupby('Day')['Sales'].agg(['mean'])
p4 = data.groupby('Day')['Price'].agg(['mean'])
#p2.iloc[i-1][0]*100 / p2.sum()[0]
ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plt.axis('off')
for i in range(1,8):
    ax.text(i-1, i-1, 'Average Sales {}: {:0f}%'.format(list(data.groupby('Day').groups.keys())[i-1], p2.iloc[i-1][0]*100 / p2.sum()[0]), color = 'white', weight = 'bold')
