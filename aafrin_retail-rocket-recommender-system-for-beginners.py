# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
items1 = pd.read_csv('../input/item_properties_part1.csv')

items2 = pd.read_csv('../input/item_properties_part2.csv')

items = pd.concat([items1, items2])

items.head(10)
items.shape
import datetime

times =[]

for i in items['timestamp']:

    times.append(datetime.datetime.fromtimestamp(i//1000.0)) 
items['timestamp'] = times
items.head(10)
#loading the event dataset



events = pd.read_csv('../input/events.csv')
events.head(10)
events.shape
print(events['event'].value_counts())

sns.countplot(x= 'event', data=events, palette="pastel")
data = events.event.value_counts()

labels = data.index

sizes = data.values

explode = (0, 0.15, 0.15)  # explode 1st slice

plt.subplots(figsize=(8,8))

# Plot

plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=False, startangle=0)

 

plt.axis('equal')

plt.show()
category_tree = pd.read_csv('../input/category_tree.csv')
category_tree.head(10)
items.loc[(items.property=='categoryid')&(items.value == '1016')].sort_values('timestamp').head()
# all unique visitors

all_customers = events['visitorid'].unique()

print("Unique visitors:", all_customers.size)



# all visitors

print('Total visitors:', events['visitorid'].size)
customer_purchased = events[events.transactionid.notnull()].visitorid.unique()

customer_purchased.size
items_new = items.loc[items.property.isin(['categoryid', 'available']), :]

print("items with categoryid and available as propery:", items_new.size)

items_new.head(20)
#grouping itemid by its event type and creating list of each of them

grouped = events.groupby('event')['itemid'].apply(list)

grouped
import operator

views = grouped['view']

# creating dictionary for key value pair 

count_view ={}

#since views is a list, we will convert it into numpy array for further manipulations

views = np.array(views[:])

#counting uniques values of views of this numpy views array

unique, counts = np.unique(views, return_counts=True)

# converting unique and counts as a dictionay with key as unique and value as counts

count_view = dict(zip(unique, counts))

#sorting the dictionary

sort_count_view = sorted(count_view.items(), key = operator.itemgetter(1), reverse = True)

# keeping number of unique views on X-axis

x = [i[0] for i in sort_count_view[:7]]

# keeping count number of views on Y-axis

y = [i[1] for i in sort_count_view[:7]]

sns.barplot(x, y, order=x, palette="rocket")
addtocart = grouped['addtocart']

# creating dictionary for key value pair 

count_addtocart ={}

#since addtocart is a list, we will convert it into numpy array for further manipulations

addtocart = np.array(addtocart[:])

#counting uniques values of addtocart items of this numpy addtocart array

unique, counts = np.unique(addtocart, return_counts=True)

# converting unique and counts as a dictionay with key as unique and value as counts

count_addtocart = dict(zip(unique, counts))

#sorting the dictionary

sort_count_addtocart = sorted(count_addtocart.items(), key = operator.itemgetter(1), reverse = True)

# keeping number of unique views on X-axis

x = [i[0] for i in sort_count_addtocart[:7]]

# keeping count number of views on Y-axis

y = [i[1] for i in sort_count_addtocart[:7]]

sns.barplot(x, y, order=x, palette="pastel")
transaction = grouped['transaction']

# creating dictionary for key value pair 

count_transaction ={}

#since addtocart is a list, we will convert it into numpy array for further manipulations

transaction = np.array(transaction[:])

#counting uniques values of addtocart items of this numpy addtocart array

unique, counts = np.unique(transaction, return_counts=True)

# converting unique and counts as a dictionay with key as unique and value as counts

count_transaction = dict(zip(unique, counts))

#sorting the dictionary

sort_count_transaction = sorted(count_transaction.items(), key = operator.itemgetter(1), reverse = True)

# keeping number of unique views on X-axis

x = [i[0] for i in sort_count_transaction[:7]]

# keeping count number of views on Y-axis

y = [i[1] for i in sort_count_transaction[:7]]

sns.barplot(x, y, order=x, palette="vlag")
#analyzing 461686 itemid

events.loc[(events.itemid==461686)]
# first - lets create a list of visitors who made a purchase

customer_purchased = events[events.transactionid.notnull()].visitorid.unique()



#lets create a list of purchased items

purchased_items = []



for customer in customer_purchased:

    purchased_items.append(list(events.loc[(events.visitorid == customer) & (events.transactionid.notnull())].itemid.values))
purchased_items[:7]
def recommend_items(item_id, purchased_items):

    recommendation_list =[]

    for x in purchased_items:

        if item_id in x:

            recommendation_list +=x

    

    # remove the pass item from the list and merge the above created list

    recommendation_list = list(set(recommendation_list) - set([item_id]))

    return recommendation_list

            
recommend_items(200793, purchased_items)
events.head(5)
from lightfm import LightFM

from lightfm.evaluation import auc_score

from scipy.sparse import coo_matrix

from sklearn import preprocessing
events = events.assign(date=pd.Series(datetime.datetime.fromtimestamp(i/1000).date() for i in events.timestamp))

events = events.sort_values('date').reset_index(drop=True)

events = events[['visitorid','itemid','event', 'date']]

events.head(5)
start_date = '2015-5-3'

end_date = '2015-5-18'

fd = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()

events = events[(events.date >= fd(start_date)) & (events.date <= fd(end_date))]
split_point = np.int(np.round(events.shape[0]*0.8))

events_train = events.iloc[0:split_point]

events_test = events.iloc[split_point::]

events_test = events_test[(events_test['visitorid'].isin(events_train['visitorid'])) & (events_test['itemid'].isin(events_train['itemid']))]
id_cols=['visitorid','itemid']

trans_cat_train=dict()

trans_cat_test=dict()



for k in id_cols:

    cate_enc=preprocessing.LabelEncoder()

    trans_cat_train[k]=cate_enc.fit_transform(events_train[k].values)

    trans_cat_test[k]=cate_enc.transform(events_test[k].values)
ratings = dict()



cate_enc=preprocessing.LabelEncoder()

ratings['train'] = cate_enc.fit_transform(events_train.event)

ratings['test'] = cate_enc.transform(events_test.event)
n_users=len(np.unique(trans_cat_train['visitorid']))

n_items=len(np.unique(trans_cat_train['itemid']))
rate_matrix = dict()

rate_matrix['train'] = coo_matrix((ratings['train'], (trans_cat_train['visitorid'], trans_cat_train['itemid'])), shape=(n_users,n_items))

rate_matrix['test'] = coo_matrix((ratings['test'], (trans_cat_test['visitorid'], trans_cat_test['itemid'])), shape=(n_users,n_items))
model = LightFM(no_components=10, loss='warp')

model.fit(rate_matrix['train'], epochs=100, num_threads=8)
auc_score(model, rate_matrix['train'], num_threads=8).mean()
auc_score(model, rate_matrix['test'], num_threads=10).mean()