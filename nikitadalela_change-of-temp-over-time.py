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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib notebook

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from IPython.display import HTML

import calendar

from plotly.subplots import make_subplots

import datetime

import seaborn as sb
df = pd.read_csv('../input/daily-temperature-of-major-cities/city_temperature.csv')

df.head()

num_features = ['Month','Day','Year','AvgTemperature']

df_hist = df[num_features]

df_hist.hist(bins =20)

#Remove records with -99 temp

df=df.drop(df[df['AvgTemperature']==-99.0].index)
df.isnull().sum()
#Rempove State column

df= df.drop(['State'], axis = 1) 



# Make new Date column

df['Date']=df.apply(lambda x: datetime.date(x['Year'], x['Month'], x['Day']), axis=1)

df.head()
plt.close()

df_2 = df.groupby('Region').mean()

df_2.reset_index(inplace=True)

df_2.head()

p1=sb.barplot(x='Region',y= 'AvgTemperature',data=df_2)

p1.set_xticklabels(p1.get_xticklabels(),rotation=20,ha='right')



plt.close()

df_3 = df[df['Region']== 'Middle East']

df_3.head()

df_3['Country'].unique()

df_4 = df_3.groupby('Country').mean()

df_4.reset_index(inplace=True)

df_4.head()

p1=sb.barplot(x='Country',y= 'AvgTemperature',data=df_4)

p1.set_xticklabels(p1.get_xticklabels(),rotation=20,ha='right')
plt.close()

df_5 = df[df['Country']=='United Arab Emirates']

df_5.head()

df_5['City'].unique()

df_6 = df_5.groupby('City').sum()

df_6.reset_index(inplace=True)

df_6.head()

p1=sb.barplot(x='City',y= 'AvgTemperature',data=df_6,order=df_6.sort_values('AvgTemperature',ascending = False).City)

p1.set_xticklabels(p1.get_xticklabels(),rotation=20,ha='right')
plt.close()

plt.figure(figsize=(17,6))

df.groupby(['Region','Country'])['AvgTemperature'].max().sort_values(ascending=False).head(10).plot(kind = 'bar',rot =15)
plt.close()

plt.figure(figsize=(14,8))

#sb.set(rc={'figure.figsize':(8,6)})

g1 = sb.lineplot(x = df['Year'],y =df['AvgTemperature'],hue = df['Region'],palette=sb.color_palette("colorblind", 7))

plt.title('Change in temperature across Regions : 1995-2020')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.legend(bbox_to_anchor=(1,1), loc=2)

plt.show()
plt.close()

#plt.figure(figsize=(17,6))

#fig, axes = plt.subplots(1,7, figsize=(12,3))

a2=[]

for r in df['Region'].unique():

    a1 =df[df['Region']==r].groupby(['Year','Month','Country'])['AvgTemperature'].max().head(1)

    a2.append(a1.to_frame())



print(a2)
c2 =[]

c3 = []

for y in df['Year'].unique():

    c1 =df[df['Year']==y].groupby(['AvgTemperature','Region']).min()['Year'].head(1)

    c2.append(c1.keys()[0])

    c3.append(str(c1.to_list()))

    



    

d = pd.DataFrame(c2,columns=['AvgTemperature','Region'])

d['Year'] =pd.Series(c3, dtype="string")



fig = px.bar(d, x='AvgTemperature', y='Year',

             hover_data=['Region'], color='AvgTemperature',

             labels={'y':'Region'},height=600)

fig.show()



#fig, axes = plt.subplots(1,7, figsize=(12,3))

c2 =[]

c3 = []

for y in df['Year'].unique():

    c1 =df[df['Year']==y].groupby(['AvgTemperature','Region']).min()['Year'].sort_values(ascending=False).head(1)

    #print(str(c1))

    c2.append(c1.keys()[0])

    #print("C2" , c2)

    c3.append(str(c1.to_list()))

    #print("C3" , c3)



    

d = pd.DataFrame(c2,columns=['AvgTemperature','Region'])

d['Year'] =pd.Series(c3, dtype="string")



fig = px.bar(d, x='AvgTemperature', y='Year',

             hover_data=['Region'], color='AvgTemperature',

             labels={'y':'Region'},height=600)

fig.show()


