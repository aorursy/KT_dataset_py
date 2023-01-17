"""
    Thanathas Chawengvorakul 6014586
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Lemonade.csv')
df.head()
avg_sale = df.Sales.mean()
avg_sale
lower_avg_salse = df.loc[df.Sales < avg_sale]
lower_avg_salse.head()
plt.scatter(x=df.Sales, y=df.Temperature)
plt.title('Sales and Temperature')
plt.xlabel('Sales')
plt.ylabel('Temperature')
day_sales  = [df.loc[df.Day == 'Sunday'].Sales.mean(), 
             df.loc[df.Day == 'Monday'].Sales.mean(),
             df.loc[df.Day == 'Tuesday'].Sales.mean(),
             df.loc[df.Day == 'Wednesday'].Sales.mean(),
             df.loc[df.Day == 'Thursday'].Sales.mean(),
             df.loc[df.Day == 'Friday'].Sales.mean(),
             df.loc[df.Day == 'Saturday'].Sales.mean()]

# atplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)[source]
plt.barh(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], day_sales)

for i, v in enumerate(day_sales):
    plt.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

