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
events = pd.read_csv('/kaggle/input/datatest/events.csv')

events.columns

events
events.event.unique()
events_by_events = events.groupby('event')
datasets = []

dataLength = []
datasets = []

dataLength = []

for i, group in events_by_events:

    # print(group.isna().sum())

    group.dropna(how="all", axis = 1, inplace=True)

    group.drop(columns=["event"], axis=1, inplace=True)

    row = {"event": i, "total Rows": group.shape[0]}

    dataLength.append(row)

    print(row)

    datasets.append({"event": i, "group": group})
datalen_df = pd.DataFrame(dataLength)

datalen_df.index = datalen_df.event

datalen_df.sort_values(by=["total Rows"], inplace=True)

datalen_df.head(20)
datalen_df.plot.barh(figsize=(30,10),x = "event")
general_cols_Series = pd.Series(list(events.columns))
for i, value in enumerate(datasets):

    value["columns"] = pd.Series(list(value["group"].columns))
for i, value in enumerate(datasets):

    print(value['event'])

    print("Columnas:",list(value['group'].columns))

    print(value['group'].info(),"\n")
len(events['person'].unique())


events[events['event'] == 'visited site']['channel']
events.groupby(['person','event']).size()
user_convert = events[events['event'] == 'conversion'][['person']]

data_user_convert = pd.merge(events, user_convert, on='person')
user_converted_full_data = data_user_convert[['person','event']]
user_converted_full_data[user_converted]
user_converted_full_data.groupby('event')
import pandas as pd 





df2 = events.groupby(["person","event"]).size().reset_index().set_index("person")

events_list=[]

for element in list(df2.index.unique()):

    events_list.append( (list(df2.loc[element].index.unique())[0], list(df2.loc[element]["event"])  ) )   



events_list[0]   



conversions_list = []

for element in events_list:

    if "conversion" in element[1]:

        conversions_list.append(element)  



conversions_list[0]



conversions_indices = []

for element in conversions_list:

    conversions_indices.append(element[0])

conversions_indices[:5]  



conversions_events = df2.loc[conversions_indices].copy().reset_index()

conversions_events.columns = ["person","event","count"]

conversions_events.groupby("event").mean().reset_index().sort_values(by = "count",ascending = False).plot(kind = "bar",y="count" , x="event",title = "Events for converted cusotmers")



df3 = df2.copy()

df3 = df3.reset_index()

boolean = ~df3["person"].isin(conversions_indices) 

notconversions_events = df3[boolean]

notconversions_events.columns = ["person","event","count"]

notconversions_events.groupby("event").mean().reset_index().sort_values(by = "count",ascending = False).plot(kind = "bar",y="count" , x="event",title = "Events for not converted cusotmers")
events[events.event == "visited site"].head()

events[events.event == "visited site"].shape

notcolumns = ["url" ,"sku",                         

"model",                       

"condition",                   

"storage",                    

"color",                       

"skus",                        

"search_term",                 

"staticpage",                  

"campaign_source",             

"search_engine"]

columns = [ x for x in list(events.columns) if x not in notcolumns ]

visited_site = events[columns]

visited_site = visited_site[visited_site.event == "visited site"]

visited_site["browser"] = visited_site['browser_version'].apply(lambda x: x.split(" ")[0])

#visited_site["browser"].value_counts().plot.barh(figsize=(30,10),logx=True)

visited_site[visited_site['new_vs_returning'] == 'New']['person'].value_counts().sum()/visited_site[visited_site['new_vs_returning'] == 'New']['person'].shape[0]