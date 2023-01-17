import pandas as pd

import numpy as np

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
foodFactData = pd.read_csv('../input/FoodFacts.csv')
foodFactData.info()
plt.figure(figsize=(12,8))

foodFactData.countries.value_counts(normalize=True).head(10).plot(kind='bar')
foodFactData.countries=foodFactData.countries.str.lower()

foodFactData.loc[foodFactData['countries'] == 'en:fr', 'countries'] = 'france'

foodFactData.loc[foodFactData['countries'] == 'en:es', 'countries'] = 'spain'

foodFactData.loc[foodFactData['countries'] == 'en:gb', 'countries']='united kingdom'

foodFactData.loc[foodFactData['countries'] == 'en:uk', 'countries']='united kingdom'

foodFactData.loc[foodFactData['countries'] == 'holland','countries']='netherlands'

foodFactData.loc[foodFactData['countries'] == 'españa','countries']='spain'

foodFactData.loc[foodFactData['countries'] == 'us','countries']='united states'

foodFactData.loc[foodFactData['countries'] == 'en:us','countries']='united states'

foodFactData.loc[foodFactData['countries'] == 'usa','countries']='united states'

foodFactData.loc[foodFactData['countries'] == 'en:cn','countries']='canada'

foodFactData.loc[foodFactData['countries'] == 'en:au','countries']='australia'

foodFactData.loc[foodFactData['countries'] == 'en:de','countries']='germany'

foodFactData.loc[foodFactData['countries'] == 'deutschland','countries']='germany'

foodFactData.loc[foodFactData['countries'] == 'en:cn','countries']='china'

foodFactData.loc[foodFactData['countries'] == 'en:be','countries']='belgium'

foodFactData.loc[foodFactData['countries'] == 'en:ch','countries']='switzerland'
plt.figure(figsize=(10,4))

foodFactData.countries.value_counts(normalize=True).head(10).plot(kind='bar')
top_countries = []

top_countries = ['france', 'united kingdom', 'spain', 'germany', 'united states']
top_countries
df = foodFactData[foodFactData.countries.isin(top_countries)]
df = df[df.trans_fat_100g.notnull()]
df.head()
trans_fat = df.groupby(['countries']).mean().trans_fat_100g
trans_fat
trans_fat = pd.DataFrame(trans_fat)
trans_fat.head()
trans_fat.sort(columns='trans_fat_100g', ascending=False, inplace=True)



trans_fat.head()
plt.figure(figsize=(12,8))

graph = trans_fat['trans_fat_100g'].plot(kind='bar')

graph.set_xlabel('countries')

graph.set_ylabel('trans_fat_100g')
tf = foodFactData.reindex(columns=['product_name', 'brands','countries','trans_fat_100g'])



tf.head()
has_tf = tf[(tf['trans_fat_100g'] > 0) & (tf['countries'] == 'united states')]
has_tf.head(15)
has_tf.sort_values('trans_fat_100g',ascending=False, inplace=True)



has_tf.head(15)