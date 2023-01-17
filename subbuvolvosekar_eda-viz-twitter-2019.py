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
# Libraries

import pandas as pd

import plotly.express as px



pd.set_option('display.max_rows', 5000)

pd.set_option('display.max_columns', 500)
data = pd.read_csv('/kaggle/input/100-mostfollowed-twitter-accounts-as-of-dec2019/Most_followed_Twitter_acounts_2019.csv')
data.shape
data.dtypes
# Renaming Column

data = data.rename(columns = {'Nationality/headquarters':'Nationality','twitter handle':'Twitterhandle'})

data.columns
data['Followers'] = data['Followers'].str.replace(',','') 

data['Followers'] = pd.to_numeric(data['Followers'])

data['Following'] = data['Following'].str.replace(',','') 

data['Following'] = pd.to_numeric(data['Following'])

data['Tweets'] = data['Tweets'].str.replace(',','') 

data['Tweets'] = pd.to_numeric(data['Tweets'])

data.dtypes
data['Industry'].unique()
data['Industry'] = data['Industry'].str.title()

data['Industry'].unique()
data['Activity'].value_counts()
data['Activity'] = data['Activity'].str.strip()

data['Activity'] = data['Activity'].str.capitalize()

data['Activity'].value_counts()
ind_cnt = data['Industry'].value_counts().reset_index()

ind_cnt = ind_cnt.rename(columns = {"Industry":"Count","index":"Industry"})

ind_cnt
fig = px.bar(ind_cnt, x="Count", y="Industry", color='Industry', orientation='h',

             hover_data=["Count", "Industry"],

             height=500,

             title='Top Industries')

fig.show()
nat_cnt = data['Nationality'].value_counts().reset_index()

nat_cnt = nat_cnt.rename(columns = {"Nationality":"Count","index":"Nationality"})

nat_cnt
fig = px.bar(nat_cnt, x="Nationality", y="Count", color='Nationality',

             hover_data=["Count", "Nationality"],

             height=600,

             title='Top Nationalities')

fig.show()
act_cnt = data['Activity'].value_counts().reset_index()

act_cnt = act_cnt.rename(columns = {"Activity":"Count","index":"Activity"})

act_cnt[0:10]
fig = px.bar(act_cnt[0:10], x="Count", y="Activity", color='Activity', orientation = 'h',

             hover_data=["Count", "Activity"],

             height=600,

             title='Top Professions')

fig.show()
twt_cnt = data.groupby('Name')['Tweets'].max().reset_index()

twt_cnt.sort_values(by=['Tweets'], inplace = True, ascending = False)

twt_cnt = twt_cnt.rename(columns = {"Tweets":"Count","index":"Tweets"})

twt_cnt[0:10]
fig = px.bar(twt_cnt[0:10], x="Count", y="Name", color='Count', orientation = 'h',

             hover_data=["Count", "Name"],

             height=600,

             title='Highest Tweets by Name')

fig.show()
fig = px.scatter(data, x="Followers", y="Following", color="Followers",

                 size='Followers', hover_data=['Name'])

fig.show()
import plotly.graph_objects as go

fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=data['Name'], y=data['Followers']/1000,

                    mode='lines+markers',

                    name='Followers'))

fig.add_trace(go.Scatter(x=data['Name'], y=data['Following']/10,

                    mode='lines+markers',

                    name='Following'))

fig.add_trace(go.Scatter(x=data['Name'], y=data['Tweets']/10,

                    mode='lines+markers',

                    name='Tweets'))



fig.show()

import plotly.graph_objects as go

fig = go.Figure()



fig = px.bar(data, x="Followers", y="Industry", color='Nationality', orientation='h',

             hover_data=["Name"],

             height=600,

             title='Twitter Celebrities')       



fig.show()

cnt_high = data.groupby(['Nationality','Name'])['Followers'].max().reset_index()

cnt_high.sort_values(by=['Nationality','Followers'], inplace = True, ascending =False)

cnt_high