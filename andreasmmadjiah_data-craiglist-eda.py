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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use('ggplot') # theme for plot's result
data=pd.read_csv("/kaggle/input/craigslist-carstrucks-data/craigslistVehiclesFull.csv")

# Data2 used for comparison when dealing with missing values

data2=pd.read_csv("/kaggle/input/craigslist-carstrucks-data/craigslistVehiclesFull.csv")

data.head(5)
data.info()
sum_null=data.isnull().sum()

sum_null=pd.DataFrame(sum_null,columns=['null'])

j=1

sum_tot=len(data)

sum_null['percent']=sum_null['null']/sum_tot

round(sum_null,3).sort_values('percent',ascending=False)
data=data.drop(columns=['size',"vin"],axis=1)
numer=data.columns[[2,3,9,16,17,18,20,23]]

data_numer=data[numer]

data_categ=data.drop(columns=numer,axis=1)

categ=data_categ.columns

sum_null.loc[numer,:].sort_values(by='percent',ascending=False)
null_numer=sum_null.loc[numer,:][sum_null['percent']>0].index

data[null_numer]=data[null_numer].fillna(data[null_numer].median())

data[null_numer].isnull().sum()
data[numer].head(20)
data_numer=data[null_numer]

data_numer['state_fips'].value_counts()

data_numer=data_numer.astype('int64')

for i in data_numer.columns:

    if data_numer[i][data_numer[i]==0].count()>0:

        print('There are %d zero values in %s' %(data_numer[i][data_numer[i]==0].count(),i))

data_numer['odometer']=data_numer['odometer'].replace(0,data_numer['odometer'].median())

data['odometer']=data_numer['odometer']

data['odometer'][data['odometer']==0]
#create correlation with hitmap



#create correlation

corr = data[numer].corr(method = 'pearson')



#convert correlation to numpy array

mask = np.array(corr)



#to mask the repetitive value for each pair

mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots(figsize = (20,5))

fig.set_size_inches(20,10)

sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)

plt.show()

data=data.drop(columns='county_fips',axis=1)

data2=data2.drop(columns='county_fips',axis=1)

numer=numer.drop('county_fips')

null_numer=null_numer.drop('county_fips')
f=plt.figure(figsize=(20,5))

f.add_subplot(1,2,1)

sns.distplot(data['odometer'], kde = True, color = 'darkblue', label = 'odometer').set_title('Distribution Plot of odometer')

f.add_subplot(1,2,2)

sns.distplot(data2['odometer'][data2['odometer'].notnull()], kde = True, color = 'darkblue', label = 'odometer').set_title('Distribution Plot of odometer')



plt.show()

f=plt.figure(figsize=(10,10))

f.add_subplot(1,2,1)

plt.scatter(data['lat'],data['weather'])

f.add_subplot(1,2,2)

plt.scatter(data2['lat'],data2['weather'])

            

plt.show()
data[null_numer]=data[null_numer].astype('int64')
f=plt.figure(figsize=(25,25))

j=1

for i in numer[[5,6,1]]:

    f.add_subplot(3,1,j)

    sns.countplot(data[i],order=data[i].value_counts().index)

    if i=='year':  

        plt.xticks(rotation=90)

    plt.xticks(fontsize=10)

    j+=1

    

plt.show()
f=plt.figure(figsize=(25,25))

j=1

for i in numer[[0,5,6,1]]:

    f.add_subplot(4,1,j)

    sns.distplot(data[i])

    if i=='year':  

        plt.xticks(rotation=90)

    plt.xticks(fontsize=10)

    j+=1

    

plt.show()
round(data['price'].describe(),2)
data['price'][data['price']> 1.499900e+04].count()
price=data['price'][data['price']<=round(data['price'].describe(),2)[-2]]

price.describe()
sns.distplot(price)

plt.show()
sns.distplot(data['price'][data['price']>150000])

plt.show()
data['year'][data['year']<1900].sort_values(ascending=False)
data['year'].describe()
year=data['year'][data['year']>=data['year'].describe()[-4]]

year.describe().astype('int64')
f=plt.figure(figsize=(20,10))

f.add_subplot(1,2,1)

sns.distplot(year.astype('int'))

f.add_subplot(1,2,2)

sns.countplot(year.astype('int'),order=year.value_counts().index)

plt.show()
f=plt.figure(figsize=(20,10))

price_year=data[['price','year']]

price_year=price_year[price_year['year']>=1900]

price_year=price_year[price_year['price']<150000]

p_y=price_year.groupby('year').mean()

p_y.reset_index(level=0,inplace=True)

f.add_subplot(2,1,1)

plt.bar(p_y['year'].astype('int'),p_y['price'],color='green')

plt.xticks(rotation=90,fontsize=10)

f.add_subplot(2,1,2)

plt.plot(p_y['year'].astype('int'),p_y['price'],color='green')

plt.xticks(rotation=90,fontsize=10)

plt.show()
p_y.sort_values('price',ascending=False).head(20)
data_categ.head(10)
data_categ=data[categ]

data_categ=data_categ.fillna(data_categ.mode().loc[0,:])

data[categ]=data_categ
data.isnull().sum()
categ
city=data['city'].value_counts()

city

x=range(470)

plt.bar(x,city)



plt.show()





data[categ].describe()
manufac=data['manufacturer'].value_counts().head(15)

plt.figure(figsize=(10,5))

plt.bar(manufac.index,manufac)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)





make=data['make'].value_counts().head(30)

plt.figure(figsize=(10,5))

sns.barplot(make.index,make)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

make=data['state_code'].value_counts()

plt.figure(figsize=(20,5))

sns.barplot(make.index,make)

plt.xticks(rotation=90,fontsize=17)

plt.yticks(fontsize=15)

make=data['state_name'].value_counts()

plt.figure(figsize=(20,5))

sns.barplot(make.index,make)

plt.xticks(rotation=90,fontsize=17)

plt.yticks(fontsize=15)

f=plt.figure(figsize=(10,30))

j=1

for i in categ[4:11]:

    f.add_subplot(7,1,j)

    sns.countplot(data[i],order=data[i].value_counts().index)

    j+=1

plt.show()
box=data[categ[4:11]]

box['price']=data['price']

box=box[box['price']<150000]

j=1    

f=plt.figure(figsize=(20,30))

for i in categ[4:8]:

    

    f.add_subplot(4,1,j)

    sns.boxplot(box[i],box['price'])

    j+=1

plt.show()
j=1    

f=plt.figure(figsize=(20,25))

for i in categ[8:11]:

    

    f.add_subplot(3,1,j)

    sns.violinplot(box[i],box['price'])

    j+=1

plt.show()
citman=data[['state_name','manufacturer','price']]

citman=citman[citman['price']<=150000]



f=plt.figure(figsize=(20,20))

f.add_subplot(2,1,1)

manu=citman[['manufacturer','price']].groupby('manufacturer').mean().sort_values('price',ascending=False).head(30)

manu.reset_index(level=0,inplace=True)

plt.bar(manu['manufacturer'],manu['price'])

plt.xticks(rotation=90,fontsize=15)



f.add_subplot(2,1,2)

state=citman[['state_name','price']].groupby('state_name').mean().sort_values('price',ascending=False).head(30)

state.reset_index(level=0,inplace=True)

plt.bar(state['state_name'],state['price'])

plt.xticks(rotation=90,fontsize=15)

plt.subplots_adjust(hspace = 0.5)



plt.show()



try2=data[['type','fuel','price']][data['price']<=150000].groupby(['type','fuel']).mean()

try2.reset_index(level=0,inplace=True)

try2.reset_index(level=0,inplace=True)

try2=try2.sort_values(['type','fuel'])

plt.figure(figsize=(15,10))

sns.barplot(x='type', y='price', hue='fuel', data=try2)

plt.xticks(rotation=90,fontsize=20)

plt.ylabel('Returns',fontsize=20)

plt.legend(fontsize=20)

plt.title('Price of type of cars for each type of fuel');