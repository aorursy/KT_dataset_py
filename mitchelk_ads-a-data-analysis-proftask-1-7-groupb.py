import numpy as np

import datetime

import pandas as pd

import pylab as pyl



import matplotlib as mpl

from matplotlib import pyplot as plt



import seaborn as sns

sns.set()



%matplotlib inline 
df1 = pd.read_csv('../input/informadataset/ds-1.csv')

df2 = pd.read_csv('../input/informadataset/ds-2.csv')

df3 = pd.read_csv('../input/informadataset/ds-3.csv')

df4 = pd.read_csv('../input/informadataset/ds-4.csv')

df5 = pd.read_csv('../input/informadataset/ds-5.csv')

df6 = pd.read_csv('../input/informadataset/ds-6.csv')

df7 = pd.read_csv('../input/informadataset/ds-7.csv')

df8 = pd.read_csv('../input/informadataset/ds-8.csv')

df9 = pd.read_csv('../input/informadataset/ds-9.csv')
frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9]

df_all = pd.concat(frames)



df_all.columns = ['Delivery date','Delivery time','Pharmacy number','Pharmacy postcode','YOB','Gender','CNK','Product name','ATC code','Unit','Price','Contribution']
df_gend = (df_all['Gender'] == 0 & 3)

df_all.drop (df_all[df_gend].index, inplace=True)

df_all.reset_index(drop=True,inplace=True)
df_all['Gender'].value_counts()
df_all['Gender'].replace([1], 'Male', inplace = True)

df_all['Gender'].replace([2], 'Female', inplace = True)
df_all['Gender'].value_counts()
df_all['Delivery time'].replace('?', np.NaN)

df_all['Delivery time'].value_counts()
df_all['DateTime'] = pd.to_datetime(df_all['Delivery date'] + ' ' + df_all['Delivery time'])

df_all['DateTime'].value_counts()
df_all.head(5)
df_all['date_offset'] = (df_all['DateTime'].dt.month * 100 + df_all['DateTime'].dt.day - 320) % 1300

df_all['Season'] = pd.cut(df_all['date_offset'], [0, 300, 602, 900, 1300], labels = ['Spring', 'summer', 'autumn', 'winter'])

df_all['Season'].value_counts()

df_all.tail(5)
df_all['Unit'].replace([0], 1, inplace = True)

df_all['Sales'] = df_all['Unit'] * df_all['Price']
df_all.head(10)
df_products = df_all.groupby(['Price','Product name'])['Gender'].count().reset_index()



df_products.columns = ['Price','Product name','Counts']

df_products.sort_values(by = 'Counts', ascending = False, inplace = True)
df_products.head(10)
for i in range(0, len(df_products)):

    if ('honorarium' in df_products.loc[i, 'Product name'].lower()): 

        df_products.loc[i, 'ProductType'] = 'Honorarium'

    elif ('tabl' in df_products.loc[i, 'Product name'].lower()): 

        df_products.loc[i, 'ProductType'] = 'Tablet'

    else:

        df_products.loc[i, 'ProductType'] = 'Pills'
df_products = df_products.sort_values(by = 'Product name')

df_products.head(5)
productsTypes = df_products.groupby("ProductType")["Product name"].count()

productsTypes.sort_values()
df_by_postcode = df_all.groupby("Pharmacy postcode")["Product name"].count()

df_by_postcode.sort_values()
df_jak_filter = df_all[df_all['Pharmacy postcode'] == 10]

df_nguyen_filter = df_all[df_all['Pharmacy postcode'] == 20].reset_index(drop=True)

df_mitchel_filter = df_all[df_all['Pharmacy postcode'] == 22]

df_nikita_filter = df_all[df_all['Pharmacy postcode'] == 30]

df_mailinh_filter = df_all[df_all['Pharmacy postcode'] == 40]
df_nguyen_filter.shape
df_s = df_nguyen_filter.sample(1000).reset_index(drop=True)

df_s.head(2)
for i in range(0, len(df_s)):

    if ('honorarium' in df_s.loc[i, 'Product name'].lower()): 

        df_s.loc[i, 'ProductType'] = 'Honorarium'

    elif ('tabl' in df_s.loc[i, 'Product name'].lower()): 

        df_s.loc[i, 'ProductType'] = 'Tablet'

    else:

        df_s.loc[i, 'ProductType'] = 'Pills'

        

df_s.head()
plt.figure(figsize=(10,4))

sns.barplot(x='ProductType', y='Price', data=df_s);
df_price_top10 = df_s.sort_values(by='Price',ascending=False).head(10)
df_price_top10.set_index('Product name',inplace=True)

dfp=df_price_top10[['Price','Contribution']]

dfp.plot.bar()
ax=plt.figure(figsize=(10,10))

sns.scatterplot(x='Price', y='Contribution', hue='Gender',data=df_s);
m_yob = df_nguyen_filter[df_nguyen_filter['Gender']=='Male']

f_yob = df_nguyen_filter[df_nguyen_filter['Gender']=='Female']



sns.set_style("darkgrid")



ax = sns.distplot(m_yob[['YOB']], bins = 120)

sns.distplot(f_yob[['YOB']], bins = 120)



plt.legend(['Male','Female','Neutral']);

plt.title('Year of Birth base on Gender')

plt.xlabel('Year of Birth')

plt.xlim(1910, 2020)
df_sales=df_s.groupby(['DateTime'])['Price'].sum().reset_index()

df_sales.set_index('DateTime',inplace=True)

df_sales.head()
fig, ax = plt.subplots(2, 1,figsize=(15,10))

df_sales['Price'].plot(ax=ax[0],color="#4C72B0")

ax[0].set_title("Sales")

ax[0].legend(["area 20"])



df_sales['rp']=df_sales['Price'].rolling(10).mean()

df_sales['rp'].plot(ax=ax[1],color="#4C72B0")

ax[1].set_title("Rolling Sales")

ax[1].legend(["area 20"]);
df_prices = df_mitchel_filter.groupby("Season")["Product name"].count()

df_prices.sort_values()
plt.figure(figsize = (10,4))

sns.countplot(x = 'Season', data = df_mitchel_filter);
YearCount = df_mitchel_filter.groupby(['Gender', 'YOB'])['Season'].count().reset_index()



YearCount.columns = ['Gender', 'YOB', 'Count']

YearCount.sort_values(by = 'Count', ascending = False, inplace = True)



YearCount.head(10)
sns.lineplot(x = 'YOB', y = 'Count', hue = 'Gender', style = 'Gender', data = YearCount)

plt.show()
ax = sns.lineplot(x = 'YOB', y = 'Count', hue = 'Gender', style = 'Gender', data = YearCount)

ax.set_xlim(1900,2020)

plt.show()
ax = plt.figure(figsize = (20, 20))

g = sns.FacetGrid(df_mitchel_filter, col = 'Season', row = 'Gender' , hue = 'Gender')

g.map(sns.regplot, 'Contribution', 'Price', fit_reg = False, x_jitter = .1)

g.add_legend()
df_nikita_filter.head(10)
df_nikita_filter = df_nikita_filter[df_nikita_filter['Pharmacy postcode'] == 30]



df_postcode = df_nikita_filter.groupby("Gender")["Product name"].count()

df_postcode
from pandas import DataFrame

datagender = {'Gender': ['Female','Male','Neutral'],

        'Total': [969246,661576,11148]}

  

dfgender = DataFrame(datagender,columns=['Gender','Total'])

dfgender.plot(x ='Gender', y='Total', kind = 'bar')

plt.title('Gender sold medicines')
df_season = df_nikita_filter.groupby(['Season'])['Product name'].count().reset_index()



df_season.columns=['Season', 'Total Product']

df_season.sort_values(by='Total Product',ascending=False,inplace=True)

df_season
df_season = df_nikita_filter.groupby(['Season'])['Product name'].count().reset_index()



df_season.columns=['Season', 'Total Product']

df_season.sort_values(by='Total Product',ascending=False,inplace=True)

df_season



bars = df_season['Season']

height = df_season['Total Product']



y_pos = np.arange(len(bars))



plt.barh(y_pos, height)

plt.yticks(y_pos, bars)

plt.title('seasons by sold products')

plt.show()
df_products = df_nikita_filter.groupby(['Price','Product name'])['Gender'].count().reset_index()



df_products.columns=['Price','Product name','Counts']

df_products.sort_values(by='Counts',ascending=False,inplace=True)

df_products10 = df_products.head(10)

df_products10
df_products = df_nikita_filter.groupby(['Price','Product name'])['Gender'].count().reset_index()



df_products.columns=['Price','Product name','Counts']

df_products.sort_values(by='Counts',ascending=False,inplace=True)

df_products10 = df_products.head(10)



bars = df_products10['Product name']

height = df_products10['Counts']



y_pos = np.arange(len(bars))



plt.barh(y_pos, height)

plt.yticks(y_pos, bars)

plt.title('most sold products')

plt.show()
df_mailinh_filter.head(10)

df_by_gender = df_mailinh_filter.groupby("Gender")["Product name"].count().reset_index()

df_by_gender.columns=['Gender', 'Total']

df_by_gender.sort_values(by='Total',ascending=False,inplace=True)



plt.figure(figsize=(15,8))

plt.xticks(rotation = 0)

plt.title('Amount of sold medicines to gender')

states_plot = sns.barplot(x=df_by_gender['Gender'],y=df_by_gender['Total'], palette="GnBu_r")

plt.show()
## I want to see which age group by which gender will most likely buy medications more 

## source: https://seaborn.pydata.org/generated/seaborn.countplot.html

## source: https://seaborn.pydata.org/generated/seaborn.distplot.html

import seaborn as sns



m_yob = df_mailinh_filter[df_mailinh_filter['Gender']=='Male']

f_yob = df_mailinh_filter[df_mailinh_filter['Gender']=='Female']



sns.set_style("darkgrid")



ax = sns.distplot(m_yob[['YOB']], bins = 120)

sns.distplot(f_yob[['YOB']], bins = 120)



plt.legend(['Male','Female','Neutral']);

plt.title('Year of Birth of patients')

plt.xlabel('Year of Birth')

plt.xlim(1910, 2020)
## Here we want to see the top 5 most sold products in region 40 

df_products = df_mailinh_filter.groupby(['Price','Product name'])['Gender'].count().reset_index()



df_products.columns=['Price','Product name','Counts']

df_products.sort_values(by='Counts',ascending=False,inplace=True)

df_products5 = df_products.head(5)

df_products5
df_products = df_mailinh_filter.groupby(['Price','Product name'])['Gender'].count().reset_index()



df_products.columns=['Price','Product name','Counts']

df_products.sort_values(by='Counts',ascending=False,inplace=True)

df_products5 = df_products.head(10)



bars = df_products5['Product name']

height = df_products5['Counts']



y_pos = np.arange(len(bars))



plt.barh(y_pos, height)

plt.yticks(y_pos, bars)

plt.title('Top 10 most sold products')

plt.show()
df_productstype = df_mailinh_filter.groupby(['Season','Product name'])['Gender'].count().reset_index()

df_productstype.columns=['Season','Product name','Counts']

df_productstype.sort_values(by='Counts',ascending=False,inplace=True)



for i in range(0, len(df_productstype)):

    if ('honorarium' in df_productstype.loc[i, 'Product name'].lower()): 

        df_productstype.loc[i, 'ProductType'] = 'Honorarium'

    elif ('tab' in df_productstype.loc[i, 'Product name'].lower()): 

        df_productstype.loc[i, 'ProductType'] = 'Tablet'

    else:

        df_productstype.loc[i, 'ProductType'] = 'Pills'

plt.figure(figsize=(10,4))

plt.title("Products type")

sns.barplot(x='ProductType', y='Counts', data=df_productstype);
df_season = df_mailinh_filter.groupby(['Season'])['Product name'].count().reset_index()



df_season.columns=['Season', 'Total Product']

df_season.sort_values(by='Total Product',ascending=False,inplace=True)

df_season_totalproduct = df_season['Total Product']



plt.figure(figsize=(15,8))

plt.xticks(rotation = 0)

plt.title('Amount of sold medicines to gender')

states_plot = sns.barplot(x=df_season['Season'],y=df_season_totalproduct, palette="GnBu_r")

plt.show()
df_clinic = df_mailinh_filter.groupby("Pharmacy number")["Product name"].count().reset_index()

df_clinic.columns=['Pharmacy number','Total of sold products']

df_clinic.sort_values(by='Total of sold products',ascending=False,inplace=True)



df_clinic.head(5)
df_clinic_sales = df_mailinh_filter.groupby("Pharmacy number")['Sales'].sum().reset_index()

df_clinic_sales.columns = ['Phamarcy number', 'Total amount of sales']

df_clinic_sales.sort_values(by='Total amount of sales',ascending=False,inplace=True)

df_clinic_sales = df_clinic_sales.head(10)



plt.figure(figsize=(15,8))

plt.xticks(rotation = 0)

plt.title('Top 10 clinics with most sales')

states_plot = sns.barplot(x=df_clinic_sales['Phamarcy number'],y=df_clinic_sales['Total amount of sales'], palette="GnBu_r")

plt.show()