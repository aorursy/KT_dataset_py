# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_json("../input/train.json")

test = pd.read_json("../input/test.json")

print (train.head())
train['source']='train'

test['source']='test'

print(train.head())

print(test.info())
print(train['bathrooms'].value_counts())

# There are houses with 0.0 bathrooms and some with floating point no of bathrooms

print(train['bedrooms'].value_counts())

# There are 9475 houses with 0 bedrooms

train['bathrooms']=train['bathrooms'].astype(int)

test['bathrooms']=test['bathrooms'].astype(int)

# if no of bathrooms are greater than 5 interest level is low else varies

# so we can create dummies

train.loc[train['bathrooms']==0, 'interest_level'] = 'low'

sns.violinplot(x='interest_level', y='bathrooms', data=train)

plt.xlabel('Interest level')

plt.ylabel('bathrooms')

plt.show()

# if 0 or greater than 4 bathrooms interest_level is low
# same patern as bathrooms only different at 0

# pattern between bedrooms and  bathrooms

print(train.loc[train['bathrooms']==0, 'bedrooms'].value_counts())

print(train.loc[train['bathrooms']==6, 'bedrooms'].value_counts())

# we can combine 6 or more #bathrooms with 5 or more bedrooms

sns.violinplot(x='interest_level', y='bedrooms', data=train)

plt.xlabel('Interest level')

plt.ylabel('bedrooms')

plt.show()
train["price_t"] =train["price"]/train["bedrooms"]

test["price_t"] = test["price"]/test["bedrooms"] 



train["room_sum"] =train["bedrooms"] + train["bathrooms"]

test["room_sum"] = test["bedrooms"] + test["bathrooms"]



train['price_per_room'] = train['price']/train['room_sum']

test['price_per_room'] = test['price']/test['room_sum']
train["created"] = pd.to_datetime(train["created"])

train["created_year"] = train["created"].dt.year

train["created_month"] = train["created"].dt.month

train["created_day"] = train["created"].dt.day

train["created_hour"] = train["created"].dt.hour

test["created"] = pd.to_datetime(test["created"])

test["created_year"] = test["created"].dt.year

test["created_month"] = test["created"].dt.month

test["created_day"] = test["created"].dt.day

test["created_hour"] = test["created"].dt.hour
plt.scatter(range(train.shape[0]), np.sort(train.price.values))

plt.xlabel('index')

plt.ylabel('price')

plt.show()

# there are outliners

ulimit = np.percentile(train.price.values, 99)

train['price'].ix[train['price']>ulimit] = ulimit

# price is right skewed so using log to create a gaussian pattern

train['price']=np.log1p(train['price'])

test['price']=np.log1p(test['price'])



plt.figure(figsize=(8,6))

sns.distplot(train.price.values, bins=50, kde=True)

plt.xlabel('price')

plt.show()

sns.violinplot(data=train,x = 'interest_level',y='price')

plt.show()
from sklearn.preprocessing import LabelEncoder

display_count = train.groupby('display_address')['display_address'].count()

plt.hist(display_count.values, bins=100, log=True, alpha=0.9)

plt.xlabel('Number of times display_address appeared', fontsize=12)

plt.ylabel('log of Count', fontsize=12)

plt.show()

# there are too many values and none of them are more than 500

# most of the values are less than 10

#so we label encode the values

address = ["display_address", "street_address"]

for x in address:

    le = LabelEncoder()

    le.fit(list(df[x].values))

    df[x] = le.transform(list(df[x].values))
train["pos"] = train.longitude.round(3).astype(str) + '_' + train.latitude.round(3).astype(str)

test["pos"] = test.longitude.round(3).astype(str) + '_' + test.latitude.round(3).astype(str)



train["density"] = train['pos'].apply(lambda x: dvals.get(x, vals.min()))

test["density"] = test['pos'].apply(lambda x: dvals.get(x, vals.min()))
print(len(train['manager_id'].unique()))

# 3481 unique managers

temp = train.groupby('manager_id').count().iloc[:,-1]

temp2 = test.groupby('manager_id').count().iloc[:,-1]

df_managers = pd.concat([temp,temp2],axis=1,join='outer')

df_managers.columns=['train_count','test_count']

print(df_managers.sort_values(by = 'train_count',ascending = False).head())

# considering only those manager_ids which are in train

man_list = df_managers['train_count'].sort_values(ascending = False).head(3481).index

ixes = df.manager_id.isin(man_list)

df10 = df[ixes][['manager_id','interest_level']]

# create dummies of interest levels

interest_dummies = pd.get_dummies(df10.interest_level)

df10 = pd.concat([df10,interest_dummies[['low','medium','high']]], axis = 1).drop('interest_level', axis = 1)

print(df10.head())

gby = pd.concat([df10.groupby('manager_id').mean(),df10.groupby('manager_id').count()], axis = 1).iloc[:,:-2]

gby.columns = ['low','medium','high','count']

gby.sort_values(by = 'count', ascending = False).head(10)

gby['manager_skill'] = gby['medium']*1 + gby['high']*2 

gby['manager_id']=gby.index

print(gby.head(5))

print(gby.shape)

df = df.merge(gby[['manager_id','manager_skill']],on='manager_id',how='outer',right_index=False)

df['manager_skill']=df['manager_skill'].fillna(0)

print(df.head())
index=list(range(train.shape[0]))

random.shuffle(index)

a=[np.nan]*len(train)

b=[np.nan]*len(train)

c=[np.nan]*len(train)



for i in range(5):

    building_level={}

    for j in train['manager_id'].values:

        building_level[j]=[0,0,0]

    

    test_index=index[int((i*train.shape[0])/5):int(((i+1)*train.shape[0])/5)]

    train_index=list(set(index).difference(test_index))

    

    for j in train_index:

        temp=train.iloc[j]

        if temp['interest_level']=='low':

            building_level[temp['manager_id']][0]+=1

        if temp['interest_level']=='medium':

            building_level[temp['manager_id']][1]+=1

        if temp['interest_level']=='high':

            building_level[temp['manager_id']][2]+=1

            

    for j in test_index:

        temp=train.iloc[j]

        if sum(building_level[temp['manager_id']])!=0:

            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])

            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])

            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])

            

train['manager_level_low']=a

train['manager_level_medium']=b

train['manager_level_high']=c



a=[]

b=[]

c=[]

building_level={}

for j in train['manager_id'].values:

    building_level[j]=[0,0,0]



for j in range(train.shape[0]):

    temp=train.iloc[j]

    if temp['interest_level']=='low':

        building_level[temp['manager_id']][0]+=1

    if temp['interest_level']=='medium':

        building_level[temp['manager_id']][1]+=1

    if temp['interest_level']=='high':

        building_level[temp['manager_id']][2]+=1



for i in test['manager_id'].values:

    if i not in building_level.keys():

        a.append(np.nan)

        b.append(np.nan)

        c.append(np.nan)

    else:

        a.append(building_level[i][0]*1.0/sum(building_level[i]))

        b.append(building_level[i][1]*1.0/sum(building_level[i]))

        c.append(building_level[i][2]*1.0/sum(building_level[i]))

test['manager_level_low']=a

test['manager_level_medium']=b

test['manager_level_high']=c
train["num_photos"] = train["photos"].apply(len)

test["num_photos"] = test["photos"].apply(len)



train["num_features"] = train["features"].apply(len)

test["num_features"] = test["features"].apply(len)



train["num_description_words"] = train["description"].apply(lambda x: len(x.split(" ")))

test["num_description_words"] = test["description"].apply(lambda x: len(x.split(" ")))
train['features'] = train["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

test['features'] = test["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))



tfidf = CountVectorizer(stop_words='english', max_features=200)

tr_sparse = tfidf.fit_transform(train["features"])

te_sparse = tfidf.transform(test["features"])