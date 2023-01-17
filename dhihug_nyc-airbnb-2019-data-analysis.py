# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

#check the shape of the datasetn, we have 48895 rows and 16 columns
df.shape


#check the data types
df.dtypes
#check for duplicates. It looks like that we don't have any duplicates
df.duplicated().sum()
#look for missing values. We have missing values in the 'name', 'host_name', 'last_review '& 'reviews_per_month' columns.
df.isnull().sum()
#drop unnecessary columns. For example, for ethical resaons we have to drom the 'host_name'column.
df.drop(['name','host_name','last_review'], axis=1, inplace=True)

#view the remaining list of columns
df.columns

#replace NaN values. In our case we are going to replace them with zeros.
df.fillna({'reviews_per_month':0}, inplace=True)

#check the results
df.reviews_per_month.isnull().sum()
#list of the unique neighbourhood_groups. In this data set we have 5 unique neighbourhood groups.

df.neighbourhood_group.unique()
#list of the unique room_types. We have 3 types of rooms. 
df.room_type.unique()
#number of different neighborhoods.We have 221 neighbourhood sub groups. 

len(df.neighbourhood.unique())


#let's see the top 10 hosts has the most bookings in this dataset. Our top host had 327 bookings in 2019. 

top_host=df.host_id.value_counts().head(10)
top_host

#create a new dataframe for tophosts
top_host_df =pd.DataFrame(top_host)
top_host_df.reset_index(inplace=True)
top_host_df.rename(columns = {'index': 'host_id', 'host_id':'bookings' }, inplace=True)
top_host_df

#plotting a barchart with Matplotlib
plt.figure()

top_host_df.plot.bar(x ='host_id', y='bookings', rot=0, color='green')
plt.xlabel('Host ID')
plt.ylabel('Number of bookings')
plt.title('Number of bookings per host - Top 10')
plt.xticks(rotation='vertical', size=10)

plt.legend('')


plt.show()


#bookings per room type : Entire homes/apt has the most number of bookings
room_types = df.room_type.value_counts()
room_types
#create a new DataFrame for the bookings by type of rooms
room_type_df = pd.DataFrame(room_types)

#rename columns
room_type_df.reset_index( inplace=True)
room_type_df.rename({'room_type':'bookings'}, axis=1, inplace=True)


room_type_df.rename(columns ={'index':'room_type'},inplace =True)
room_type_df

#plot a pie chart to show the distrubtion of the % of bookings by room type.

labels = room_type_df.room_type
numbers = room_type_df.bookings

pie1 = plt.figure(figsize=(10,7))
plt.pie(numbers, labels=labels,autopct='%1.1f%%', startangle=90)
plt.title('% of booking by room type, NYC 2019', weight='bold')

plt.show()

#a bar chart by Seaborn to show the bookings by type of rooms in the 5 neighbourhood groups
plt.figure(figsize=(10,8))
ax = sns.countplot(df['neighbourhood_group'],hue=df['room_type'])
#number of bookings by the neighbourhood groups. 
df.neighbourhood_group.value_counts()
#bar graph to show the number of bookings by neighbourhood groups
plt.figure(figsize=(8,4))
ax = sns.countplot(df["neighbourhood_group"])
#top 10 neighbourhoods sub groups
df.neighbourhood.value_counts().head(10)
# bar graph using Matplotlib to the the top 10 neighbourhood sub groups

top_10 = df.neighbourhood.value_counts().head(10)
plt.figure(figsize=(8, 4))
x = list(top_10.index)
y = list(top_10.values)
x.reverse()
y.reverse()

plt.title('Most Popular Neighbourhoods', size=14)
plt.ylabel('Number of hosts in this area')
plt.xlabel('Neighbourhood Area ')
plt.xticks( rotation='vertical', size=11)

plt.bar(x, y , color='green')
#create a filter to further analyse the data from the top 5 neighbourhoods

#the top 5 neighbourhoods -Williamsburg,Bedford-Stuyvesant,Harlem, Bushwick & Upperwest Side- are situated either in Manhattan or Brooklyn. 

filt =df.loc[(df['neighbourhood_group'] == 'Manhattan') & (df['neighbourhood_group'] == 'Brooklyn') &(df['neighbourhood_group'] == 'Queens')
             &(df['neighbourhood_group'] == 'Bronx')& (df['neighbourhood_group'] == 'Staten Island')
             
             |(df['neighbourhood'] == 'Williamsburg') | (df['neighbourhood'] == 'Bedford-Stuyvesant')|(df['neighbourhood'] == 'Harlem')|(df['neighbourhood'] == 'Bushwick')| (df['neighbourhood'] == 'Upper West Side')]       

filt.shape

#check whether we've applied the right filter
df.neighbourhood.value_counts().head(5).sum()
#create a new dataframe for the top 5 neighbourhoods _ all of them in either Brooklyn or Manhattan

new_df = pd.DataFrame(filt)
new_df.drop(['calculated_host_listings_count', 'availability_365','id','latitude','longitude', 'reviews_per_month'], axis=1, inplace=True)

#comparison of the top 2 neighbourhood groups, Brooklyn and Manhattan

#get the number of bookings by neighbourhood group for the 2 top groups
count_man=new_df.loc[df['neighbourhood_group'] == 'Manhattan'].host_id.count()
count_brook=new_df.loc[df['neighbourhood_group'] != 'Manhattan'].host_id.count()

mean_pr_man=new_df.loc[df['neighbourhood_group'] == 'Manhattan'].price.mean()
mean_pr_brook=new_df.loc[df['neighbourhood_group'] != 'Manhattan'].price.mean()



test = [count_brook, mean_pr_brook, count_man,mean_pr_man]  
# number of bookings in Brooklyn is more than double than that of Manhattan
test
#create a dew dataframe that shows the number of bookings and average price/night in the top 2 neighbourhood groups
d = {'name': ['Brooklyn', 'Manhattan'],'mean_price': [116.11, 158.12], 'bookings': [4629, 10099]}

test_df = pd.DataFrame(d)
test_df.set_index('name', inplace=True)
test_df
#a double bar chart to show the number of bookings and average price/night in Manhattan and Brooklyn 

fig = plt.figure(figsize=(10,5))
test_df.plot.bar( secondary_y= 'bookings', label = 'Name')

ax1, ax2 = plt.gcf().get_axes()
ax1.set_ylabel('Average price/night')
ax2.set_ylabel('Number of bookings')

plt.show()
#lets compare the average prices in the 5 neighbourhood_groups

neighb_mean= df.groupby('neighbourhood_group').mean()

plt.figure(figsize =(8,8))

neighbourhood_group = [neighbourhood_group for neighbourhood_group, df in df.groupby('neighbourhood_group')]

plt.bar(neighbourhood_group, neighb_mean['price'], color='green')
plt.xticks(neighbourhood_group, rotation='vertical', size=10)
plt.xlabel('Neighbourhood Group')
plt.title('Average price per neighbourhood group' , size=16)
plt.ylabel('Average price/night')
plt.show()