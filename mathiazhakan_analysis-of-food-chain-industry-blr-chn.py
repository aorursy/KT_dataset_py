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

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

zomato = pd.read_csv("../input/zomato-restaurants-in-india/zomato_restaurants_in_India.csv")
#checking for the recodrs and features

print('Number of Features:',zomato.shape[1])

print('Number of Records:',zomato.shape[0])
#Checking for Duplicated values

zomato[zomato.duplicated()].count()[0]
zomato.columns
redundant=['highlights','rating_text','res_id','url','address','city_id','country_id','zipcode','longitude','latitude','currency','photo_count','delivery','takeaway','locality_verbose','timings','opentable_support']
print('Number of Features before Pruning:',zomato.shape[1])

zomato.drop(redundant, axis=1, inplace=True)

print('Number of Features after Pruning:',zomato.shape[1])

print('Number of Hotels before Removal of Duplicates:',zomato.shape[0])

zomato.drop_duplicates(inplace=True)

print('Number of Hotels after Removal of Duplicates:',zomato.shape[0])
zomato.head()
zc=zomato[(zomato['city']=='Chennai')|(zomato['city']=='Bangalore')]

zc.to_csv('zomato_chn_blr.csv')

import pandas as pd

zom_comp = pd.read_csv("../input/test-data/zomato_chn_blr.csv")
zom_comp.head(2)
zom_comp.drop('Unnamed: 0', axis=1, inplace=True)

zom_comp.head(2)
## checking for null values

zom_comp.isnull().sum()
# Removing unwanted characters from establishment column

zom_comp['establishment']=zom_comp['establishment'].apply(lambda x :str(x).replace('[',''))

zom_comp['establishment']=zom_comp['establishment'].apply(lambda x :str(x).replace(']',''))

zom_comp['establishment']=zom_comp['establishment'].apply(lambda x :str(x).replace("'",''))

#Creating a New feature for better understanding of ratings

l=[]

for i in range(0,zom_comp.shape[0]):

    if zom_comp.iloc[i,7]<=1:

        l.append('Poor')

    elif zom_comp.iloc[i,7]>1 and zom_comp.iloc[i,7]<=2:

        l.append('Average')

    elif zom_comp.iloc[i,7]>2 and zom_comp.iloc[i,7]<=3:

        l.append('Good')

    elif zom_comp.iloc[i,7]>3 and zom_comp.iloc[i,7]<=4:

        l.append('Very Good')

    elif zom_comp.iloc[i,7]>4 and zom_comp.iloc[i,7]<=5:

        l.append('Excellent')

        

rat=pd.Series(l, name='Word_rating')

#concating with dataframe

zom_comp=pd.concat([zom_comp,rat], axis=1,join='outer')
# Naming the Price_ratings

dic={1:'Low',2:'Average',3:'High',4:'Very High'}

zom_comp['price_type']=zom_comp['price_range'].map(dic)
zom_comp.isnull().sum()

# Feature creation has not affected out data_set
zom_comp.loc[:,['Word_rating','price_type']]
fig,ax=plt.subplots(1,1,figsize=(6,6))

fig.suptitle('Number of Restaurants', fontsize=15)

zom_comp.groupby('city')['cuisines'].count().plot(kind='bar',color = 'orange', ax=ax)

for i in range(2):

    plt.text(x = i-0.08 , y=zom_comp.groupby('city')['cuisines'].count()[i]+15, s = zom_comp.groupby('city')['cuisines'].count()[i], size =15)

chn=zom_comp[zom_comp['city']=='Chennai']

blr=zom_comp[zom_comp['city']=='Bangalore']

a=len(list(chn.locality.unique()))

b=len(list(blr.locality.unique()))

v=[]

v.append(b)

v.append(a)

fig,ax=plt.subplots(1,1,figsize=(6,6))

fig.suptitle('Number of Localities', fontsize=15)

sns.barplot(x=zom_comp.city.unique(), y =v, ax=ax)

for i in range(2):

    plt.text(x = i-0.08 , y=v[i]+1, s = v[i], size =15)

chn_10=(zom_comp[zom_comp['city']=='Chennai'])['locality'].value_counts().head(10)

blr_10=(zom_comp[zom_comp['city']=='Bangalore'])['locality'].value_counts().head(10)

fig,ax=plt.subplots(1,2,figsize=(30,8))

a=sns.barplot(chn_10.index,chn_10.values, ax=ax[0])

a.set_xlabel('Locality')

a.set_ylabel('Count')

b=sns.barplot(blr_10.index,blr_10.values, ax=ax[1])

b.set_xlabel('Locality')

b.set_ylabel('Count')

fig.suptitle('Major localities of hotels', fontsize=20)

a.title.set_text('Chennai')

b.title.set_text('Bangalore')

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

for i in range(10):

    a.text(x = i-0.08 , y=list(chn_10.values)[i]+1, s = list(chn_10.values)[i], size =15)

    b.text(x = i-0.08 , y=list(blr_10.values)[i]+1, s = list(blr_10.values)[i], size =15)



print('Major Localities of Chennai \n',chn_10)  

print('Major Localities of Bangalore \n',blr_10)    
chn_10_typ=(zom_comp[zom_comp['city']=='Chennai'])['establishment'].value_counts().head(10)

blr_10_typ=(zom_comp[zom_comp['city']=='Bangalore'])['establishment'].value_counts().head(10)
fig,ax=plt.subplots(1,2,figsize=(10,5))

c=sns.barplot(chn_10_typ.index,chn_10_typ.values, ax=ax[0])

c.set_xlabel('Type of Restaurant')

c.set_ylabel('Count')

d=sns.barplot(blr_10_typ.index,blr_10_typ.values, ax=ax[1])

d.set_xlabel('Type of Restauran')

d.set_ylabel('Count')

c.title.set_text('Major Types of hotels in chennai')

d.title.set_text('Major Types of hotels in bangalore')

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

for i in range(10):

    c.text(x = i-0.4 , y=list(chn_10_typ.values)[i]+1, s = list(chn_10_typ.values)[i], size =10)

    d.text(x = i-0.4 , y=list(blr_10_typ.values)[i]+1, s = list(blr_10_typ.values)[i], size =10)



pd.crosstab(zom_comp['city'], zom_comp['establishment']).loc[['Chennai'],list(chn_10_typ.index)]

pd.crosstab(zom_comp['city'], zom_comp['establishment']).loc[['Bangalore',],list(blr_10_typ.index)]

chn_10_brands=(zom_comp[(zom_comp['city']=='Chennai')])['name'].value_counts().head(10)

blr_10_brands=(zom_comp[(zom_comp['city']=='Bangalore')])['name'].value_counts().head(10)

fig,ax=plt.subplots(1,2, figsize=(10,5))

e=sns.barplot(chn_10_brands.index,chn_10_brands.values, ax=ax[0])

f=sns.barplot(blr_10_brands.index,blr_10_brands.values, ax=ax[1])

fig.suptitle('Top Brands based on Number of Outlets', fontsize=20)

e.title.set_text('Chennai')

f.title.set_text('Bangalore')

e.set_xlabel('Brand')

f.set_xlabel('Brand')

e.set_ylabel('Number of Outlets')

f.set_ylabel('Number of Outlets')



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

for i in range(10):

    e.text(x=i-0.3, y=list(chn_10_brands.values)[i]+0.5, s=list(chn_10_brands.values)[i])

    f.text(x=i-0.3, y=list(blr_10_brands.values)[i]+0.5, s=list(blr_10_brands.values)[i])



    

    
chn_loc_est=pd.crosstab(zom_comp['locality'], zom_comp['establishment']).loc[list(chn_10.index),list(chn_10_typ.index)]

blr_loc_est=pd.crosstab(zom_comp['locality'], zom_comp['establishment']).loc[list(blr_10.index),list(blr_10_typ.index)]

fig,ax=plt.subplots(1,2, figsize=(20,10))

g=chn_loc_est.plot(kind='bar',stacked=True, ax=ax[0])

h=blr_loc_est.plot(kind='bar',stacked=True,ax=ax[1])

plt.legend()

fig.suptitle('Spread of Major Types of restaurants in each of the Top 10 Localities in both the cities', fontsize=20)

mcc=zom_comp[(zom_comp['city']=='Chennai')]['cuisines'].value_counts().head(10)

mcb=zom_comp[(zom_comp['city']=='Bangalore')]['cuisines'].value_counts().head(10)

fig,ax=plt.subplots(1,2, figsize=(10,5))

l=sns.barplot(mcc.index,mcc.values, ax=ax[0])

m=sns.barplot(mcb.index,mcb.values, ax=ax[1])

fig.suptitle('Major cuisines', fontsize=20)

l.title.set_text('Chennai')

m.title.set_text('Bangalore')

l.set_xlabel('cuisines')

m.set_xlabel('cuisines')

l.set_ylabel('Number of each cuisines')

m.set_ylabel('Number of each cuisines')



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

for i in range(10):

    l.text(x=i-0.3, y=list(mcc.values)[i]+0.5, s=list(mcc.values)[i])

    m.text(x=i-0.3, y=list(mcb.values)[i]+0.5, s=list(mcb.values)[i])



    

    
avg_chn=pd.DataFrame(chn.groupby('establishment')['average_cost_for_two'].median().sort_values(ascending=False))

avg_blr=pd.DataFrame(blr.groupby('establishment')['average_cost_for_two'].median().sort_values(ascending=False))

av_c=avg_chn.loc[chn_10_typ.index].sort_values(by=['average_cost_for_two'],ascending=False)

av_b=avg_blr.loc[blr_10_typ.index].sort_values(by=['average_cost_for_two'],ascending=False)



print(av_c)

print(av_b)
fig,ax=plt.subplots(1,2, figsize=(20,7))

avc=av_c.plot(kind='bar',ax=ax[0])

avb=av_b.plot(kind='bar',ax=ax[1])

fig.suptitle('Average price',fontsize=20)

avc.title.set_text('Chennai')

avb.title.set_text('Bangalore')

avc.set_xlabel('Establishment Type')

avb.set_xlabel('Establishment Type')

avc.set_ylabel('Price')

avb.set_ylabel('price')

avc.legend('')

avb.legend('')

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

for i in range(10):

    avc.text(x=i-0.3, y=list(av_c.values)[i]+30, s=list(av_c.values)[i])

    avb.text(x=i-0.3, y=list(av_b.values)[i]+30, s=list(av_b.values)[i])



chn_pri_loc=pd.crosstab(zom_comp['locality'], zom_comp['price_type']).loc[list(chn_10.index)]

blr_pri_loc=pd.crosstab(zom_comp['locality'], zom_comp['price_type']).loc[list(blr_10.index)]

chn_pri_loc

blr_pri_loc
fig,ax=plt.subplots(1,2, figsize=(10,5))

n=chn_pri_loc.plot(kind='bar',stacked=True, ax=ax[0])

o=blr_pri_loc.plot(kind='bar',stacked=True, ax=ax[1])

fig.suptitle('Price Range in Top classified Localities',fontsize=20)

n.title.set_text('Chennai')

o.title.set_text('Bangalore')

n.set_xlabel('Locality')

o.set_xlabel('Locality')

n.set_ylabel('count')

o.set_ylabel('count')

n.legend(bbox_to_anchor=[-0.15,1])

o.legend(bbox_to_anchor=[1.5, 1])



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

chn_pri_est=pd.crosstab(chn['establishment'], chn['price_type']).loc[list(chn_10_typ.index),:]

blr_pri_est=pd.crosstab(blr['establishment'], blr['price_type']).loc[list(blr_10_typ.index),:]

chn_pri_est
blr_pri_est
fig,ax=plt.subplots(1,2, figsize=(10,5))

n=chn_pri_est.plot(kind='bar',stacked=True, ax=ax[0])

o=blr_pri_est.plot(kind='bar',stacked=True, ax=ax[1])

fig.suptitle('Price Range in Top classified Restaurants',fontsize=20)

n.title.set_text('Chennai')

o.title.set_text('Bangalore')

n.set_xlabel('Restaurant Type')

o.set_xlabel('Restaurant Type')

n.set_ylabel('count')

o.set_ylabel('count')

n.legend(bbox_to_anchor=[-0.15,1])

o.legend(bbox_to_anchor=[1.5, 1])



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

print('Number of Low priced hotels in chennai:',chn_pri_est['Low'].sum())

print('Number of Low priced hotels in Bangalore:',blr_pri_est['Low'].sum())
fig,ax=plt.subplots(1,2, figsize=(10,5))

n=chn_pri_est.sum().plot(kind='bar', ax=ax[0])

o=blr_pri_est.sum().plot(kind='bar', ax=ax[1])

fig.suptitle('Price Range in Top classified Restaurants',fontsize=20)

n.title.set_text('Chennai')

o.title.set_text('Bangalore')

n.set_xlabel('Restaurant Type')

o.set_xlabel('Restaurant Type')

n.set_ylabel('count')

o.set_ylabel('count')

n.legend(bbox_to_anchor=[-0.15,1])

o.legend(bbox_to_anchor=[1.5, 1])

n.legend('')

o.legend('')

chl=list(chn_pri_est.sum().values)

bl=list(blr_pri_est.sum().values)

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

for i in range(4):

    n.text(x=i-0.15, y=chl[i]+10, s=chl[i])

    o.text(x=i-0.15, y=bl[i]+10, s=bl[i])
chn_qb=chn[(chn['establishment']=='Quick Bites')]

chn_qb=chn_qb.sort_values(by=['aggregate_rating','votes'], ascending=False).head(10)

chn_qb['rank']=chn_qb['votes'].rank(ascending=False,method='dense')
chn_qb.sort_values(by=['rank'])
blr_qb=blr[(blr['establishment']=='Quick Bites')]

blr_qb=blr_qb.sort_values(by=['aggregate_rating','votes'], ascending=False).head(10)

blr_qb['rank']=blr_qb['votes'].rank(ascending=False,method='dense')

blr_qb.sort_values(by=['rank'])
fig,ax=plt.subplots(1,2, figsize=(20,5))

sns.barplot(chn_qb['name'], chn_qb['price_range'],ax=ax[0],hue=chn_qb['Word_rating'])

sns.barplot(blr_qb['name'], blr_qb['price_range'],ax=ax[1],hue=blr_qb['Word_rating'])

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)
# Top Casual dinings in chennai

chn_cd=chn[(chn['establishment']=='Casual Dining')]

chn_cd=chn_cd.sort_values(by=['aggregate_rating','votes'], ascending=False).head(10)

chn_cd['rank']=chn_cd['votes'].rank(ascending=False,method='dense')

chn_cd.sort_values(by=['rank'])



#Top casual dinings in bangalore

blr_cd=blr[(blr['establishment']=='Casual Dining')]

blr_cd=blr_cd.sort_values(by=['aggregate_rating','votes'], ascending=False).head(10)

blr_cd['rank']=blr_cd['votes'].rank(ascending=False,method='dense')

fig,ax=plt.subplots(1,2, figsize=(20,5))

sns.barplot(chn_cd['name'], chn_cd['price_range'],ax=ax[0],hue=chn_cd['Word_rating'])

sns.barplot(blr_cd['name'], blr_cd['price_range'],ax=ax[1],hue=blr_cd['Word_rating'])

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)
fig,ax=plt.subplots(1,2, figsize=(10,5))

a1=sns.countplot(chn_cd['name'], ax=ax[0],hue=chn_cd['price_range'])

b1=sns.countplot(blr_cd['name'], ax=ax[1],hue=blr_cd['price_range'])

fig.suptitle('Top Casual Dinings',fontsize=20)

a1.title.set_text('Chennai')

b1.title.set_text('Bangalore')

a1.set_xlabel('Restaurant Name')

b1.set_xlabel('Restaurant Name')

a1.set_ylabel('count')

b1.set_ylabel('count')

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)



print(chn_cd.groupby('name')['name'].count())

print(blr_cd.groupby('name')['name'].count())