import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings as war

war.filterwarnings('ignore')



import os

print(os.listdir('../input/zomato-hyd-mini-data-set/'))
data=pd.read_csv('../input/zomato-hyd-mini-data-set/zomo.csv',encoding='latin-1')

data.head()
data.duplicated(['zomo_t','zomo_res_name']).sum()
data=data.rename(columns={'zomo_t':'Name','Avege_cost':'Average_cost','votes':'Votes','place':'Location'})
data.head()
data.drop(['zomo_res_name'],axis=1,inplace=True)
data.head()
data.drop(['zomo_rating'],axis=1,inplace=True)
data.head()
data['Average_cost']=data['Average_cost'].apply(lambda x : x.split()[0])

data['Average_cost']=data['Average_cost'].apply(lambda x: x.split('?')[1])

data['Average_cost']=data['Average_cost'].apply(lambda x : x.replace(',',''))

data['Average_cost']=data['Average_cost'].astype('int')
print('The maximum Average cost for Two person:',data['Average_cost'].max())

print('The minimum Average cost for Two person:',data['Average_cost'].min())

print('Top 10 Restaurent with Maximum number of votes')

top10_votes=data.sort_values(by='Votes',ascending=False).head(10)

top10_votes
print('Top 10 restaurent with higest votes sorted by price')

top10_votes.sort_values(by='Average_cost')
plt.figure(figsize=(10,6))

g=sns.countplot(data['Location'],palette='Set1')

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha='right')

g