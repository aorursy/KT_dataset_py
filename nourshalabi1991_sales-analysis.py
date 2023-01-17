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
import numpy as np

from numpy.random import randn

import pandas as pd

from pandas import Series, DataFrame



import matplotlib.pyplot as plt

from pylab import rcParams
address = '/kaggle/input/superstore-sales/superstore.csv'



df = pd.read_csv(address, index_col='Order Date', parse_dates=True)

df.head()
df['Quantity'].plot()
df2 = df.sample(n=100, random_state=30, axis=0)



plt.xlabel('Order Date')

plt.ylabel('Order Quantity')

plt.title('Sales Summary')



df2['Quantity'].plot()
df2['Quantity'].plot(kind='hist')
plt.hist(df2['Quantity'])

plt.plot()
import seaborn as sb

sb.set_style('whitegrid')
sb.distplot(df2['Profit'])
# A Scatter Plot is used to show the relationship between two variables



''''Features :

1 - Trend 

2 - Scatter 

3 - Outlier

'''



df2.plot(kind='scatter', x='Sales', y='Profit', c=['darkgray'], s=150)
sb.regplot(x='Sales', y='Profit', data=df2, scatter=True)
sb.pairplot(df)
sb.pairplot(df2)
df_subset = df[['Sales', 'Quantity', 'Discount', 'Profit']]

sb.pairplot(df_subset)

plt.show()
sb.regplot(x='Discount', y='Profit', data=df2, scatter=True)
sb.regplot(x='Discount', y='Sales', data=df2, scatter=True)
df2_subset = df2[['Sales', 'Quantity', 'Discount', 'Profit']]

sb.pairplot(df2_subset)

plt.show()
sb.boxplot(x='Discount', y='Profit', data=df2, palette='hls')

sb.boxplot(x='Quantity', y='Sales', data=df, palette='hls')
sb.boxplot(x='Quantity', y='Sales', data=df2, palette='hls')