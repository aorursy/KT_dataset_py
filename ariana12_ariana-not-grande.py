# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10,6) # define figure size of pyplot
pd.set_option("display.max_columns", 100) # set max columns when displaying pandas DataFrame

pd.set_option("display.max_rows", 200) # set max rows when displaying pandas DataFrame
df = pd.read_csv('/kaggle/input/uisummerschool/Marketing.csv')

df
#tampilkan data

df.plot(kind='box');
df.describe() # statistical description of DataFrame columns, numerical only
data_offline = pd.read_csv('/kaggle/input/uisummerschool/Offline_sales.csv')

data_offline
data_offline['Quantity'].describe()
group_by_per_tanggal = data_offline.groupby(['Quantity'])['InvoiceDate'].count() #ini cuma mengelompokkan dan menghitung saja

group_by_per_tanggal
plt.plot(group_by_per_tanggal)
group_by_per_tanggal.plot(kind='bar');
data_produk = pd.read_csv('/kaggle/input/uisummerschool/Product.csv')

data_online = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')
data = pd.merge(data_produk,data_online)

data
grouped = data.groupby(['Product','Date'])['Revenue'].count() #ini cuma mengelompokkan dan menghitung saja

#pd.crosstab(data.Product,data.AvgPrice)

grouped

pd.crosstab([df['Sex'], df['Survived']], df['Pclass'], margins=True)
#data.crosstab([data['Product'], data['Quantity']], data["Product Category (Enhanced E-commerce)"], margins=True)

import seaborn as sns

sns.barplot(x='Product SKU', y='Quantity', data=data)
data.plot(kind='line', x='Product');
data.hist();