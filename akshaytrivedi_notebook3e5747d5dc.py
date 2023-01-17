import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
events = pd.read_csv('../input/events.csv')

category_tree = pd.read_csv('../input/category_tree.csv')

items1 = pd.read_csv('../input/item_properties_part1.csv')

items2 = pd.read_csv('../input/item_properties_part2.csv')

items = pd.concat([items1, items2])
# Having a look at the values in the events dataset

events.head()
user_with_buy = dict()

for row in events.itertuples():

    if row.visitorid not in user_with_buy:

        user_with_buy[row.visitorid] = {'view':0 , 'addtocart':0, 'transaction':0};

    if row.event == 'addtocart':

        user_with_buy[row.visitorid]['addtocart'] += 1 

    elif row.event == 'transaction':

        user_with_buy[row.visitorid]['transaction'] += 1

    elif row.event == 'view':

        user_with_buy[row.visitorid]['view'] += 1 
event_to_user = pd.DataFrame(user_with_buy)
event_to_user.head()
user_to_event = event_to_user.transpose()
event_to_user.head()
len(event_to_user.loc[(dataframe['view']!=0) & (event_to_user['addtocart']==0) & (event_to_user['transaction']==0)])
len(event_to_user.loc[(event_to_user['addtocart']!=0) | (event_to_user['transaction']!=0)])
# Total number of users

len(event_to_user)
# Items bought or viewed

len(set(events['itemid']))
# All items

len(set(items['itemid']))
# Define activity as sum of (#addToCarts, #transaction, #views)

event_to_user['activity'] = event_to_user['view'] + event_to_user['addtocart'] + event_to_user['transaction']
event_to_user.describe()
# Users with just 1 view

len(event_to_user[event_to_user['activity']==1])