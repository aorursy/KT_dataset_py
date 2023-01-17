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
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel(r'/kaggle/input/miq-fashion-data/Fashion - Assignment Data (3).xlsx', header = 1)
df.head()
df.info()
df = df.sort_values(by = 'timestamp' ).reset_index()
df.timestamp
df['Revenue'] = pd.to_numeric(df['Revenue'],errors='coerce')

df.Number_of_Products.value_counts()
df['Revenue'].sum()
def sum_prods(x):
    try:
        if type(x)==int:
            return x
        else:
            return sum(map( (lambda x : int(x)), x.split(',')))
    except:
        return 0
    
sum_prods(df.Number_of_Products[1])
sum_prods_per_order = df.Number_of_Products.map(lambda x : sum_prods(x))
df.Number_of_Products
sum_prods_per_order
df.sum_prods_per_order = sum_prods_per_order
sum(df.sum_prods_per_order)
# AVG Counts
df.sum_prods_per_order.sum()/len(df.sum_prods_per_order)
df.Revenue.sum()/len(df.Revenue)
df.head()
df.City.nunique()
top_10_cities = df[['Revenue', 'City']].groupby('City').sum().sort_values(by = 'Revenue', ascending =  False)[:10]
top_10_cities
plt.figure(figsize = (12,7))
sns.barplot(x = top_10_cities.index , y = top_10_cities.Revenue) 
plt.show()
df.Country_Province.unique()
top_provs = df[['Revenue', 'Country_Province']].groupby('Country_Province').sum().sort_values(by = 'Revenue', ascending =  False)
top_provs
plt.figure(figsize = (10,6))
sns.barplot(x = top_provs.index , y = top_provs.Revenue) 
plt.show()
df.Payment_Type.unique()
top_pymttype = df[['Revenue', 'Payment_Type']].groupby('Payment_Type').sum().sort_values(by = 'Revenue', ascending =  False)
top_pymttype
plt.figure(figsize = (8,5))
sns.barplot(x = top_pymttype.index , y = top_pymttype.Revenue)
plt.show()
# Frequency per count: 
df.sum_prods_per_order.value_counts()
25517/len(df.Revenue)*100
df['user ID'].value_counts()
sum(df['user ID'].value_counts().map(lambda x : x>1)), len(df['user ID'])*100-sum(df['user ID'].value_counts().map(lambda x : x>1))
sum(df['user ID'].value_counts().map(lambda x : x>1))/len(df['user ID'])*100
