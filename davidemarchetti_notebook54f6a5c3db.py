import numpy as np 
import pandas as pd 
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


dati=pd.read_csv('../input/groceries-dataset/Groceries_dataset.csv')
dati.head(10)
dati.info()
dati.shape
dati.nunique()
dati['Date']=pd.to_datetime(dati['Date'], format='%d-%m-%Y')
dati['Year']=dati['Date'].dt.year
dati['Month']=dati['Date'].dt.month
dati['DayofWeek']=dati['Date'].dt.dayofweek
dati
dati.itemDescription.value_counts()[:10]
sns.countplot(data=dati, y='itemDescription', order=dati['itemDescription'].value_counts()[:10].index)
dati.itemDescription.value_counts()[-10:]
dati.Member_number.value_counts()[:10]
dati.groupby('Month').Month.count()

sns.countplot(data=dati, x='Month', order=dati['Month'].value_counts().index)
sns.countplot(data=dati, x='DayofWeek', order=dati['DayofWeek'].value_counts().index)
products=dati['itemDescription'].unique()
one_hot=pd.get_dummies(dati['itemDescription'])
dati.drop(columns=(['itemDescription', 'Year', 'Month', 'DayofWeek']), inplace=True)
dati=dati.join(one_hot)
basket=dati.groupby(['Member_number', 'Date'])[products[:]].apply(sum)
basket = basket.reset_index()[products]
def encode(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
        
basket=basket.applymap(encode)
frequent_itemsets = apriori(basket, min_support=0.003, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
rules.sort_values(by='lift', ascending=False).head(10)
