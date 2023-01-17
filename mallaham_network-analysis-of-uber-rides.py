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
#importing all related libraries

import pandas as pd

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/My Uber Drives - 2016.csv")

data.head(5)

data = data.drop(data.index[len(data)-1])
data.keys()
data.info()
#replacing missing values with PURPOSE* equals to "other"

data['PURPOSE*'].replace(np.nan, 'Other', inplace=True)

data.info()
data['PURPOSE*'].value_counts()
#creating a dataframe containing type of purpose and the number of rides associated with each purpose

ride_count = data['PURPOSE*'].value_counts().tolist()

ride_purpose= data['PURPOSE*'].value_counts().index.tolist()

ride_info = list(zip(ride_purpose,ride_count))

ride_info = pd.DataFrame(ride_info,columns=['PURPOSE','COUNT'])



ax = sns.barplot(x='COUNT',y='PURPOSE',data=ride_info,order=ride_info['PURPOSE'].tolist())

ax.set(xlabel='Number of Rides', ylabel='Purpose')

plt.show()



data['CATEGORY*'].value_counts()
ride_summary = data.groupby('PURPOSE*').mean()

ride_summary
ride_summary['PURPOSE*']=ride_summary.index.tolist()

ax = sns.barplot(x='MILES*',y='PURPOSE*',data=ride_summary,order=ride_summary.sort_values('MILES*',ascending=False)['PURPOSE*'].tolist())

ax.set(xlabel='Avrg Miles', ylabel='Purpose')

plt.show()
start_list = [info.split(' ') for info in data['START_DATE*'].tolist()]

stop_list = [info.split(' ') for info in data['END_DATE*'].tolist()]



start_df  = pd.DataFrame(start_list,columns=['Start_Date','Start_Time'])

end_df  = pd.DataFrame(stop_list,columns=['End_Date','End_Time'])



sub_data = data[['CATEGORY*','START*','STOP*','MILES*','PURPOSE*']]

start_end_info = pd.concat([start_df,end_df,],axis=1)
rides = pd.concat([start_end_info,sub_data],axis=1)
rides.head(5)
rides.groupby('Start_Date').sum()
rides_per_month = rides.groupby('Start_Date').sum()

rides_per_month['Month']=pd.to_datetime(rides_per_month.index.tolist()) #converting dates to a python friendly format

rides_per_month['Month']= rides_per_month['Month'].dt.to_period("M") #grouping dates by month

rides_per_month= rides_per_month.sort_values(by= 'Month',ascending=True)
total_miles_per_month= rides_per_month.groupby('Month').sum()
total_miles_per_month['MONTH']=total_miles_per_month.index.tolist()

total_miles_per_month['MONTH']=total_miles_per_month['MONTH'].astype(str) #converting the time stamp format to string

ax = sns.barplot(x='MILES*',y='MONTH',data=total_miles_per_month,order=total_miles_per_month.sort_values('MONTH',ascending=False)['MONTH'].tolist())

ax.set(xlabel='Total Miles', ylabel='Month')

plt.show()
g = nx.Graph()
g= nx.from_pandas_dataframe(rides,source='START*',target='STOP*',edge_attr=['Start_Date','Start_Time','End_Date','End_Time','CATEGORY*','MILES*','PURPOSE*'])
print(nx.info(g))
plt.figure(figsize=(12,12)) 

nx.draw_circular(g,with_labels=True,node_size=100)

plt.show()
#identifying which location is being visited more frequently (whether for pickup or dropoff)

location=[]

degree=[]

for node in g:

    location.append(node)

    degree.append(g.degree(node))



degree_dist_list = list (zip(location,degree))

degree_dist = pd.DataFrame(degree_dist_list, columns=['Location','Degree'])

degree_dist.sort(columns='Degree',ascending=False)
#plotting locations that were visited at least 4 times.

ax = sns.barplot(x='Degree',y='Location',data=degree_dist[degree_dist['Degree']>=4],order=degree_dist[degree_dist['Degree']>=4].sort_values('Degree',ascending=False)['Location'].tolist())

ax.set(xlabel='Number of Rides', ylabel='Purpose')

plt.figure(figsize=(30,20))

plt.show()
#since the highest number of rides were for meeting purposes, here I visualize this network of rides

g2= nx.Graph()

g2 = nx.from_pandas_dataframe(rides[rides['PURPOSE*']=='Meeting'],source='START*',target='STOP*')
print (nx.info(g2))

plt.figure(figsize=(12,12)) 

nx.draw_shell(g2,with_labels=True,node_size=100)

plt.show()
location_2=[]

degree_2=[]

for node in g2:

    location_2.append(node)

    degree_2.append(g.degree(node))



degree_dist_list_2 = list (zip(location_2,degree_2))

degree_dist_2 = pd.DataFrame(degree_dist_list_2, columns=['Location','Degree'])

degree_dist_2.sort(columns='Degree',ascending=False)