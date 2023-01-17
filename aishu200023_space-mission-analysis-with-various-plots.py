# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import wordcloud as wc
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style='dark')
import plotly.express as px
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/all-space-missions-from-1957/Space_Corrected.csv")
df.head()
df.info()
df.describe()
df.shape
#check null values
df.isna().sum()
# Extract the launch year
df['DateTime'] = pd.to_datetime(df['Datum'])
df['Year'] = df['DateTime'].apply(lambda datetime: datetime.year)

# Extract the country of launch
df["Country"] = df["Location"].apply(lambda location: location.split(", ")[-1])

df.head(10)
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1', ' Rocket'], axis=1)
df
df['Month'] = df['DateTime'].apply(lambda datetime: datetime.month)
df
# Country vs no.of launches
country = df.groupby('Country').count()['Detail'].sort_values(ascending=False).reset_index()
country.rename(columns={"Detail":"No of Launches"},inplace=True)
country.head(10).style.background_gradient(cmap='Blues').hide_index()
#bar plot on the above for better visualization
plt.figure(figsize=(8,8))
ax = sns.countplot(y="Country", data=df, order=df["Country"].value_counts().index)
ax.set_xscale("log")
ax.axes.set_title("Country vs. # Launches (Log Scale)",fontsize=18)
ax.set_xlabel("Number of Launches (Log Scale)",fontsize=16)
ax.set_ylabel("Country",fontsize=16)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.show()
#Companies vs no.of launches by them
comp = df['Company Name'].value_counts().reset_index()
comp.columns = ['company', 'number of starts']
comp = comp.sort_values(['number of starts'])
fig = px.bar(
    comp, 
    x='number of starts', 
    y="company", 
    orientation='h', 
    title='Number of launches by every company', 
    height=1000, 
    width=800
)
fig.show()
# treemap  for company vs launches
company_list = list(df['Company Name'].unique())

num_launch = []

# get number of lunchs for each company
for n in company_list:
    num_launch.append(((df[df['Company Name']== n]).shape)[0])

#convert the lists into data dict.    
data = {'Company': company_list, 'launchs': num_launch}

#create dataframe
df_comp = pd.DataFrame(data=data, columns= ['Company', 'launchs'])
df_comp
fig = px.treemap(df_comp.sort_values(by = 'launchs', ascending= False).reset_index(drop = True),
                         path = ['Company'], values= 'launchs', height = 700,
                         title = 'Number of launchs Company wise',
                         color_discrete_sequence = px.colors.qualitative.Light24)
fig.data[0].textinfo = 'label + text+ value'

fig.show()
location_list = list(df['Location'].unique())

launch = []

for n in location_list:
    launch.append(((df[df['Location']== n]).shape)[0])

data_l = {'Company': location_list, 'launchs': launch}


df_loc = pd.DataFrame(data=data_l, columns= ['location', 'launchs'])
fig = px.pie(df_loc, values=df_loc['launchs'], names=df_loc.index,
             title='location and Their Lauches Ratio in The World',
            )
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    template='plotly_white'
)
fig.show()
#Status va launches
sts = df['Status Rocket'].value_counts().reset_index()
sts.columns = ['status', 'count']
fig = px.pie(
    sts, 
    values='count', 
    names="status", 
    title='Rocket status', 
    width=500, 
    height=500
)
fig.show()
# Mission status vs Count
plt.figure(figsize=(6,6))
ax = sns.countplot(y="Status Mission", data=df, order=df["Status Mission"].value_counts().index, palette="ocean_r")
ax.set_xscale("log")
ax.axes.set_title("Mission Status vs. Count",fontsize=18)
ax.set_xlabel("Count",fontsize=16)
ax.set_ylabel("Mission Status",fontsize=16)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.show()
#yearwise launches
date= df.groupby('Year').count()['Detail'].reset_index()
plt.figure(figsize=(20,6))
b=sns.barplot(x='Year', y='Detail', data=date)
plt.ylabel('no of launches')
plt.title(' No of launches per year')
_=b.set_xticklabels(b.get_xticklabels(), rotation=90, horizontalalignment='right')