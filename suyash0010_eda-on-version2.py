import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
df = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

df.head()
df.info()
df.floor.replace('-',0,inplace=True)
df['floor']=df.floor.astype('int')
duplicate_rows = df[df.duplicated()]
duplicate_rows.shape
df.drop_duplicates(inplace=True)
df.describe()
sns.set(font_scale=1.5)
df_int = df.select_dtypes('int')
len(df_int.columns)
fig,ax = plt.subplots(5,2,figsize=(30,60))
for i in range(10):
    sns.boxplot('city',df_int.columns[i],data=df,ax=ax[i%5,i//5])
    ax[i%5,i//5].set_title(df_int.columns[i])
df_int = df.select_dtypes('int')
z = np.abs(stats.zscore(df_int))
df = df[(z < 3).all(axis=1)]
sns.set(font_scale=1.5)
fig,ax = plt.subplots(5,2,figsize=(30,60))
for i in range(10):
    sns.boxplot('city',df_int.columns[i],data=df,ax=ax[i%5,i//5])
    ax[i%5,i//5].set_title(df_int.columns[i])
cities = pd.Series.unique(df['city'])
sns.set(font_scale=1.5)
fig,ax=plt.subplots(5,1,figsize=(20,40))
for i in range(len(cities)):
    sns.distplot(df.loc[df['city']==cities[i],'total (R$)'],ax=ax[i])
    ax[i].set_title(cities[i])
sns.set(style="whitegrid", font_scale=2.0)
df_city_fur =df.groupby(['city','furniture'])['total (R$)'].count()
df_city_furniture=df_city_fur.reset_index(level='furniture')
fig,ax = plt.subplots(1,5,figsize=(28,10))
unique_cities=pd.Series.unique(df['city'])
for i in range(len(unique_cities)):
    sns.barplot(x=df_city_furniture.loc[unique_cities[i],'furniture'].values, y=df_city_furniture.loc[unique_cities[i],'total (R$)'].values,ax=ax[i],palette='Set1')
    ax[i].set_title(unique_cities[i])
    ax[i].set(xlabel='furniture',ylabel='Number of Houses')
    plt.tight_layout()
df_city_fur_total = df.groupby(['city','furniture'])['total (R$)'].mean().round(2)
df_city_furniture_total=df_city_fur_total.reset_index(level='furniture')
fig,ax = plt.subplots(1,5,figsize=(28,10))
unique_cities=pd.Series.unique(df['city'])
for i in range(len(unique_cities)):
    sns.barplot(x=df_city_furniture_total.loc[unique_cities[i],'furniture'].values, y=df_city_furniture_total.loc[unique_cities[i],'total (R$)'].values,ax=ax[i],palette='muted')
    ax[i].set_title(unique_cities[i])
    ax[i].set(xlabel='furniture',ylabel='Average Total Price')
    plt.tight_layout()
df_city_fur_total = df.groupby(['city','furniture'])['rent amount (R$)'].mean().round(2)
df_city_furniture_total=df_city_fur_total.reset_index(level='furniture')
fig,ax = plt.subplots(1,5,figsize=(28,10))
unique_cities=pd.Series.unique(df['city'])
for i in range(len(unique_cities)):
    sns.barplot(x=df_city_furniture_total.loc[unique_cities[i],'furniture'].values, y=df_city_furniture_total.loc[unique_cities[i],'rent amount (R$)'].values,ax=ax[i],palette='Set2')
    ax[i].set_title(unique_cities[i])
    ax[i].set(xlabel='furniture',ylabel='Average Rental Price')
    plt.tight_layout()
sns.set(font_scale=2)
factors = ['rooms','bathroom','parking spaces','floor']
fig,ax = plt.subplots(len(factors),1,figsize=(20,25))
for i in range(len(factors)):
    sns.barplot(df[factors[i]],df['total (R$)'],ax=ax[i])
    plt.tight_layout()

sns.set(font_scale=1.5)
fig,ax=plt.subplots(5,1,figsize=(20,40))
for i in range(len(cities)):
    sns.scatterplot(x='area',y='total (R$)',data=df[df['city']==cities[i]],ax=ax[i])
    ax[i].set_title(cities[i])