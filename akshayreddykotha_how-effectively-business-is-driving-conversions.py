import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
events = pd.read_csv('../input/events.csv')

events.sample(10)

#events.info()

events.isnull().sum()
x = list()

x = events['event'].value_counts()

#type(x)



prop_of_events = x/ sum(x)

prop_of_events
perc_conv = x[2]*100/x[1]

perc_conv
#Define new data frame which doesn't have 'view' events

events_non_view = events[(events['event'] != "view")]

events_non_view.sample(5)
events_non_view[events_non_view['visitorid'] == 1303838]
events_non_view[(events_non_view["visitorid"] == 1210136) & (events_non_view["itemid"] == 253214)]
#An alternative way of checking addtocart and transaction logged for a user and a different item.

cond1 = (events_non_view["visitorid"] == 1210136)

cond2 = (events_non_view["itemid"] == 396732)



events_non_view[cond1 & cond2]
events.groupby('event').count()
unique_visitors = events.groupby('event')['visitorid'].nunique()

unique_visitors
len(events['visitorid'].unique())
perc_uniq_visitors = unique_visitors/ sum(unique_visitors)

print(perc_uniq_visitors)
perc_uniq_visitors[1]*100/perc_uniq_visitors[0]
#mask = (events_non_view["visitorid"] == 1210136) & events_non_view["itemid"] == 396732)

#events_non_view.ix[mask, events_non_view]

#events_non_view[events_non_view['visitorid'] == 1210136] & events_non_view[events_non_view['itemid'] == 396732]