import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
su=pd.read_csv('../input/supermarket-sales/supermarket_sales - Sheet1.csv')
su.head()
# cogs means 'Cost Of Goods Sold'.
# COGS does not include indirect expenses, like certain overhead costs. Do not factor things like utilities, marketing expenses,
# or shipping fees into the cost of goods sold.
su.drop(['Invoice ID'],axis=1, inplace=True)
su.head()
su.shape
su.describe()
sns.heatmap(su.isnull())
su.info()
sns.pairplot(su, hue='City')
fig=plt.gcf()
fig.set_size_inches(13,13)
sns.countplot(x='Gender', data=su)
plt.title('Gender',fontsize=20)
fig=plt.gcf()
fig.set_size_inches(10,5)
su["Gender"].value_counts()
sns.countplot(x='City', data=su)
plt.title('Cities',fontsize=20)
fig=plt.gcf()
fig.set_size_inches(10,5)
su["City"].value_counts()
plt.rcParams['figure.figsize'] = (15, 8)
Y = pd.crosstab(su['City'], su['Gender'])
Y.div(Y.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
plt.title('City vs Gender', fontweight = 30, fontsize = 20)
plt.legend(loc="upper right")
plt.show()
sns.countplot(x='Customer type', data=su, palette='gnuplot')
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.title('Customer type',fontsize=20)
su["Customer type"].value_counts()
sns.countplot(x='City', hue='Customer type', data=su)
plt.title('City vs Gender',fontsize=20)
sns.countplot(x='Gender', hue='Customer type', data=su)
plt.title('Gender vs Customer')

plt.subplot(2,2,1)
# plt.figure(figsize=(15,10))
dat = su[su['Gender']=='Female']
chart = sns.countplot(x = 'City', data=dat, hue='Customer type')
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')
chart.plot()
plt.title('Cities vs Customer type vs Female')

plt.subplot(2,2,2)
# plt.figure(figsize=(15,10))
dat = su[su['Gender']=='Male']
chart = sns.countplot(x = 'City', data=dat, hue='Customer type')
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')
chart.plot()
plt.title('Cities vs Customer type vs Male')
sns.countplot(x='Product line', data=su)
plt.title('Product line',fontsize=20)
# fig=plt.gcf()
# fig.set_size_inches(15,10)
su["Product line"].value_counts()
sns.countplot(x='Gender', data=su, hue='Product line', palette='gist_stern')
plt.title('Product Popularity categorised by Gender',fontsize=20)
sns.countplot(x='City', data=su, hue='Product line')
fig=plt.gcf()
fig.set_size_inches(20,10)
plt.title('Product Popularity categorised by City',fontsize=20)
plt.subplot(2,2,1)
# plt.figure(figsize=(8,8))
dat = su[su['Gender']=='Female']
chart = sns.countplot(x = 'City', data=dat, hue='Product line')
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')
chart.plot()
plt.title('Product Popularity categorised by City and only Female')

plt.subplot(2,2,2)
# plt.figure(figsize=(5,5))
dat = su[su['Gender']=='Male']
chart = sns.countplot(x = 'City', data=dat, hue='Product line')
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')
chart.plot()
plt.title('Product Popularity categorised by City and only Male')
sns.distplot(su['Total'])
plt.title('Total',fontsize=20)
g = sns.FacetGrid(su, col='City', size=4)
g.map(plt.hist, 'Total', alpha=0.5, bins=15,color='r', )
g.add_legend()
plt.show()
g = sns.FacetGrid(su, row='Gender', col='City', size=4)
# plt.title('distribution of Total categorised City and Gender ')
g.map(plt.hist, 'Total', alpha=0.5, bins=15,color='m')
g.add_legend()
plt.show()

su.groupby('City').sum()['Total']
print('city, Yangon')
dat = su[su['City']=='Yangon']
dat.groupby('Gender').sum()['Total']
print('city, Naypyitaw')
dat = su[su['City']=='Naypyitaw']
dat.groupby('Gender').sum()['Total']

print('city, Mandalay')
dat = su[su['City']=='Mandalay']
dat.groupby('Gender').sum()['Total']
su.groupby('Gender').sum()['Total']
su.groupby('Product line').sum()['Total']
sns.countplot(x='Payment', data=su)
plt.title('Payment types',fontsize=20)
fig=plt.gcf()
fig.set_size_inches(10,5)
sns.countplot(x='Payment', data=su,hue='Gender')
plt.title('Payment vs Gender', fontsize=20)
sns.countplot(x='City', data=su, hue='Payment', palette='winter_r')
plt.title('Cities vs Payment',fontsize=20)
su["Payment"].value_counts()
sns.distplot(su['Rating'])
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.subplot(2,2,1)
dat = su[su['Gender']=='Male']
chart=sns.distplot(dat['Rating'], color='g')
plt.title('Gender, Male')

plt.subplot(2,2,2)
dat = su[su['Gender']=='Female']
chart=sns.distplot(dat['Rating'], color='g')
plt.title('Gender, Female')

df1 = su[su['City']=='Yangon']
df2 = su[su['City']=='Naypyitaw']
df3 = su[su['City']=='Mandalay']

plt.subplot(3,3,1)
sns.distplot(df1['Rating'],color='indigo')
plt.title('Yangon')

plt.subplot(3,3,2)
sns.distplot(df2['Rating'],color='indigo')
plt.title('Naypyitaw')

plt.subplot(3,3,3)
sns.distplot(df3['Rating'],color='indigo')
plt.title('Mandalay')
su1=df1[df1['Gender']=='Female']
su2=df2[df2['Gender']=='Female']
su3=df3[df3['Gender']=='Female']

su_male1=df1[df1['Gender']=='Male']
su_male2=df2[df2['Gender']=='Male']
su_male3=df3[df3['Gender']=='Male']

plt.subplot(3,3,1)
plt.title('Yangon')
sns.distplot(su1['Rating'],color='r',hist=False,label="Female")
sns.distplot(su_male1['Rating'],color='b',hist=False,label="Male")


plt.subplot(3,3,2)
plt.title('Naypyitaw')
sns.distplot(su2['Rating'],color='r',hist=False,label="Female")
sns.distplot(su_male2['Rating'],color='b',hist=False,label="Male")

plt.subplot(3,3,3)
plt.title('Mandalay')
sns.distplot(su3['Rating'],color='r',hist=False,label="Female")
sns.distplot(su_male3['Rating'],color='b',hist=False,label="Male")



su['Product line'].unique()
su_pro1=df1[df1['Product line']=='Health and beauty']
su_pro2=df1[df1['Product line']=='Electronic accessories']
su_pro3=df1[df1['Product line']=='Sports and travel']
su_pro4=df1[df1['Product line']=='Food and beverages']
su_pro5=df1[df1['Product line']=='Fashion accessories']

su_pro1n=df2[df2['Product line']=='Health and beauty']
su_pro2n=df2[df2['Product line']=='Electronic accessories']
su_pro3n=df2[df2['Product line']=='Sports and travel']
su_pro4n=df2[df2['Product line']=='Food and beverages']
su_pro5n=df2[df2['Product line']=='Fashion accessories']

su_pro1m=df3[df3['Product line']=='Health and beauty']
su_pro2m=df3[df3['Product line']=='Electronic accessories']
su_pro3m=df3[df3['Product line']=='Sports and travel']
su_pro4m=df3[df3['Product line']=='Food and beverages']
su_pro5m=df3[df3['Product line']=='Fashion accessories']

plt.subplot(1,3,1)
sns.distplot(su_pro1['Rating'],color='r',hist=False,label="Health and beauty")
sns.distplot(su_pro2['Rating'],color='m',hist=False,label="Electornic accessories")
sns.distplot(su_pro3['Rating'],color='c',hist=False,label="Sports and travel")
sns.distplot(su_pro4['Rating'],color='b',hist=False,label="Food and beverages")
sns.distplot(su_pro5['Rating'],color='y',hist=False,label="Fashion accessories")
plt.title('Yangon')

plt.subplot(1,3,2)
sns.distplot(su_pro1n['Rating'],color='r',hist=False,label="Health and beauty")
sns.distplot(su_pro2n['Rating'],color='m',hist=False,label="Electornic accessories")
sns.distplot(su_pro3n['Rating'],color='c',hist=False,label="Sports and travel")
sns.distplot(su_pro4n['Rating'],color='b',hist=False,label="Food and beverages")
sns.distplot(su_pro5n['Rating'],color='y',hist=False,label="Fashion accessories")
plt.title('Naypyitaw')

plt.subplot(1,3,3)
sns.distplot(su_pro1m['Rating'],color='r',hist=False,label="Health and beauty")
sns.distplot(su_pro2m['Rating'],color='m',hist=False,label="Electornic accessories")
sns.distplot(su_pro3m['Rating'],color='c',hist=False,label="Sports and travel")
sns.distplot(su_pro4m['Rating'],color='b',hist=False,label="Food and beverages")
sns.distplot(su_pro5m['Rating'],color='y',hist=False,label="Fashion accessories")
plt.title('Mandalay')

sns.boxplot(x='Product line', y='Total', data=su)
sns.boxplot(x='City', y='Total', data=su,hue='Gender')
fig=plt.gcf()
fig.set_size_inches(10,10)
sns.boxplot(x='Product line', y='Total', data=su, hue='City')
