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
from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

import pandas as pd

data = pd.read_excel("/kaggle/input/online-retail/Online_Retail.xlsx") 

data.head(10)

#data.describe()
data.groupby(["Country"]).count()
data['Description'] = data['Description'].str.strip()

data.head()

data.shape
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)

data['InvoiceNo'] = data['InvoiceNo'].astype('str')

data = data[~data['InvoiceNo'].str.contains('C')]

data.shape
data.head()
df1 = data[data['Country'] =="France"]

df1.shape
df2=df1.groupby(['InvoiceNo', 'Description'])['Quantity'].sum()

df2
df3=df2.unstack()

df3
df3.reset_index()
basket=df1.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')

basket.head()

def encode_units(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1



basket_sets = basket.applymap(encode_units)

basket_sets.drop('POSTAGE', inplace=True, axis=1)

basket_sets.head()
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

frequent_itemsets
from mlxtend.frequent_patterns import fpgrowth



fpgrowth(basket_sets, min_support=0.07,use_colnames=True)