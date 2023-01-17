import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from collections import Counter

%matplotlib inline
data=pd.read_csv("../input/indian-food-101/indian_food.csv")
data.head()
## Check for nulls,

##since in the data description it is mentioned that missing values are indicated with -1,

for c in data.columns:

    print(f'''Total Missing values in column {c} is {len(data[data[c]=='-1'])}''')
data.isna().sum()
data[data['region'].isna()]
data.loc[data['region'].isna(),'region']='North'
data.loc[data['region']=='-1',]
data.loc[data['region']=='-1','region']='All region'
##state column,

data.loc[data['state']=='-1']
data.loc[data['state']=='-1','state']='All States'
data.loc[data['flavor_profile']=='-1',]
flavor_dict={'Chapati':'sweet',

'Naan':'sweet',

'Rongi':'sweet',

'Kanji':'sweet',

'Pachadi':'sweet',

'Paniyaram':'sweet',

'Paruppu sadam':'sour',

'Puli sadam':'sour',

'Puttu':'sweet',

'Sandige':'sweet',

'Sevai':'sweet',

'Thayir sadam':'sour',

'Theeyal':'spicy',

'Bhakri':'sweet',

'Copra paak':'sweet',

'Dahi vada':'sweet',

'Dalithoy':'spicy',

'Kansar':'sweet',

'Farsi Puri':'spicy',

'Khar':'sweet',

'Luchi':'sweet',

'Bengena Pitika':'sweet',

'Bilahi Maas':'sweet',

'Black rice':'sour',

'Brown Rice':'sweet',

'Chingri Bhape':'sweet',

'Pakhala':'spicy',

'Pani Pitha':'sweet',

'Red Rice':'spicy'}
## Using a loop to change the values.I think there will be a better way to do this !!

for c in data.loc[data['flavor_profile']=='-1',['name','flavor_profile']]['name']:

    print(f'Assigning flavor profile for {c}')

    data.loc[data['name']==c,'flavor_profile']=flavor_dict[c]
##Now lets check again,

for c in data.columns:

    print(f'''Total Missing values in column {c} is {len(data[data[c]=='-1'])}''')
## Prep time and Cook time,

data.loc[(data['prep_time']==-1) & (data['diet']=='vegetarian'),'prep_time']=10

data.loc[(data['cook_time']==-1) & (data['diet']=='vegetarian'),'cook_time']=10

data.loc[(data['prep_time']==-1) & (data['diet']=='non vegetarian'),'prep_time']=20

data.loc[(data['cook_time']==-1) & (data['diet']=='non vegetarian'),'cook_time']=20
##How many dishes ?

print(f'''There are {data['name'].nunique()} dishes ''')
(data['diet'].value_counts()/data['name'].nunique())*100
(data['course'].value_counts()/data['name'].nunique())*100
(data['flavor_profile'].value_counts()/data['name'].nunique())*100
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

sns.distplot(data['prep_time'],color='red')

plt.title('Preparation time distribution')

plt.xlabel('Preparation Time')

plt.ylabel('Frequency')

plt.subplot(1,2,2)

sns.distplot(data['cook_time'],color='blue')

plt.title('Cooking time distribution')

plt.xlabel('Cooking Time')

plt.ylabel('Frequency')
data['total_time']=data['prep_time']+data['cook_time']
plt.figure(figsize=(8,8))

sns.boxplot(x='diet',y='total_time',data=data,palette=sns.color_palette('colorblind'))

plt.title('Total Time taken for a dish by diet',fontsize=15)

plt.xlabel('Diet preference',fontsize=8)

plt.ylabel('Total Time',fontsize=8)
## Dishes with total time > 400 minutes:

data.loc[data['total_time']>=400,]
##total ingredients required:

data['total_ingredients']=data['ingredients'].apply(lambda x:len(set(x.split())))
data['total_ingredients'].describe()
data.loc[data['total_ingredients']==12,]
data.loc[data['total_ingredients']==2,]
plt.figure(figsize=(18,8))

plt.subplot(1,3,1)

sns.boxplot(x='course',y='total_ingredients',data=data,palette=sns.color_palette('colorblind'))

plt.title('Course vs total ingredients',fontsize=15)

plt.xlabel('Course',fontsize=8)

plt.ylabel('Total Ingredients',fontsize=8)

plt.subplot(1,3,2)

sns.boxplot(x='flavor_profile',y='total_ingredients',data=data,palette=sns.color_palette('colorblind'))

plt.title('Flavor Profile vs total ingredients',fontsize=15)

plt.xlabel('Flavor Profile',fontsize=8)

plt.ylabel('Total Ingredients',fontsize=8)

plt.subplot(1,3,3)

sns.boxplot(x='diet',y='total_ingredients',data=data,palette=sns.color_palette('colorblind'))

plt.title('Diet vs total ingredients',fontsize=15)

plt.xlabel('Diet',fontsize=8)

plt.ylabel('Total Ingredients',fontsize=8)
def ingre_count(d):

    foo=list(d['ingredients'].apply(lambda x:[i.strip() for i in x.split(',')]))

    return Counter(i for j in foo for i  in j).most_common(5)
## top 10 ingredients in Indian cusine,

ingre_count(data)
## top 5 in vegetarian dishes,

ingre_count(data.loc[data['diet']=='vegetarian',])
## top 5 in non-vegetarian dishes,

ingre_count(data.loc[data['diet']=='non vegetarian',])
### top 5 in spicy dishes

ingre_count(data.loc[data['flavor_profile']=='spicy',])
### top 5 in sweet dishes

ingre_count(data.loc[data['flavor_profile']=='sweet',])
### top 5 in sour dishes

ingre_count(data.loc[data['flavor_profile']=='sour',])
### top 5 in bitter dishes

ingre_count(data.loc[data['flavor_profile']=='bitter',])