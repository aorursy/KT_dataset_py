# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/stock-keeping-unit-merkle-sokrati/SKU-data-Assignment-2.csv')
data.head()
data.describe()
data = data.transpose()

data.head()
data.index.name = 'Date'

data.columns = data.iloc[0, :]

data.drop('SKU', inplace=True)
data.head()
data.index.names
data.index = pd.to_datetime(data.index)

data.head()
data['Total Sales'] = data.iloc[:,:-1].sum(axis=1)

data.head()
data['Month Year'] = data.index.strftime('%b-%Y')

data['Month Year'].head()
monthlySales = data.groupby(by= ['Month Year'])['Total Sales'].sum()

print("Monthly Sales: ")

print(monthlySales)
data['Quarter'] = data.index.to_period('Q')

data['Quarter'].head()
quarterlySales = data.groupby(by= ['Quarter'])['Total Sales'].sum()

print("Quarterly Sales: ")

print(quarterlySales)
print("Top 3 Months in overall sales: ")

monthlySales.sort_values(ascending=False)[:3]
data.columns
SKUsMonthlySales = data.groupby(by= ['Month Year'])[data.columns[:-3]].apply(lambda x : x.sum())

SKUsMonthlySales.head()
SKUsMonthlySales.shape
SKUsMonthlySales.describe()
SKUsMonthlySales.iloc[0,:].sort_values()
SKUsMonthlySales.iloc[1,:].sort_values()
SKUsMonthlySales.iloc[5,:].sort_values()
SKUsMonthlySales['98201'].describe().round(2)
SKUsMonthlySales['65001'].describe().round(2)