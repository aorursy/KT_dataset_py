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
import os

import matplotlib.pyplot as plt

import seaborn as sns

import regex as re
#Loading the datasets

primary_result = pd.read_csv('../input/us-elections-dataset/us-2016-primary-results.csv', sep=';')

primary_result.head(5)
#Checking is there are missing values

missing_values = primary_result.isnull().sum()

missing_values
primary_result.drop(columns=['fips'], inplace=True)

primary_result.head(5)
#Lets see the total number of votes casted in the primary

primary_result.votes.sum()
#Number of counts of Republicans and Democrats

primary_result['party'].value_counts()
#Let's take a look at Vermont

vermont = primary_result.loc[primary_result.state.isin(['Vermont'])]

vermont.head(5)
vermont.votes.sum()
votes = vermont.groupby('candidate')['votes'].sum().reset_index()

Votes_Vermont = votes.sort_values('votes',ascending=False)

Votes_Vermont
fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = Votes_Vermont,x = 'candidate',y = 'votes', ax = ax)

ax.set_ylabel('Count')

ax.set_title('Vermont votes')

for index,Votes_Vermont in enumerate(Votes_Vermont['votes'].astype(int)):

       ax.text(x=index-0.1 , y =Votes_Vermont+2 , s=f"{Votes_Vermont}" , fontdict=dict(fontsize=8))

plt.show()
ver_m = vermont.party.value_counts().reset_index()



import plotly.express as px

fig = px.pie(ver_m, values=ver_m['party'], names=ver_m['index'])

fig.update_layout(title = 'Party with Most Votes in Vermont')

fig.show()
primary = primary_result.groupby('candidate')['votes'].sum().reset_index()

Primary = primary.sort_values('votes',ascending=False).head(5)

Primary.head()
fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = Primary,x = 'candidate',y = 'votes', ax = ax)

ax.set_ylabel('Count')

ax.set_title('Top 5 Candidates with Most Votes')

for index,Primary in enumerate(Primary['votes'].astype(int)):

       ax.text(x=index-0.1 , y =Primary+2 , s=f"{Primary}" , fontdict=dict(fontsize=8))

plt.show()
state = primary_result.groupby('state')['votes'].sum().reset_index()

States = state.sort_values('votes',ascending=False).head(10)

States
fig = px.pie(States, values=States['votes'], names=States['state'])

fig.update_layout(title = 'Top 10 States with Most Votes')

fig.show()
county = primary_result.groupby('county')['votes'].sum().reset_index()

County = county.sort_values('votes',ascending=False).head(10)

County
plt.subplots(figsize=(10,10))

splot = sns.barplot(x=County['county'],y=County['votes'], palette = 'winter_r')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')



plt.xlabel('county',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.xticks(rotation=90)

plt.yticks(fontsize=15)

plt.title('Top 10 County Votes',fontsize=25);
Hill_cl = primary_result.loc[primary_result.candidate.isin(['Hillary Clinton'])]

Hill_cl.head(5)
state = Hill_cl.groupby(['state','state_abbreviation'])['votes'].sum().reset_index()

Hill_state_max = state.sort_values('votes',ascending=False).head(15)

Hill_state_max
import plotly.express as px



fig = px.choropleth(locations= ['CA','FL','NY','IL','TX','PA','OH','NC','MA','MI','NJ','GA','MD','VA','WI'], 

                    locationmode="USA-states", 

                    color= Hill_state_max['votes'],

                    labels={'color':'votes', 'locations':'State'},

                    scope="usa") 





fig.update_layout(

    

    title_text = 'Top 15 States with Most Votes',

    geo_scope='usa'

)

fig.show()
Hill_state_min = state.sort_values('votes',ascending=True).head(5)

Hill_state_min
fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = Hill_state_min,x = 'state',y = 'votes', ax = ax)

ax.set_ylabel('Count')

ax.set_title('Bottom 5 states with Less Votes')
hillary = Hill_cl.groupby(['state','state_abbreviation'])['fraction_votes'].mean().reset_index()

Hillary = hillary.sort_values('fraction_votes',ascending=False).head(10)

Hillary
fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = Hillary,x = 'state',y = 'fraction_votes', ax = ax)

ax.set_ylabel('Count')

ax.set_title('Top 10 states with Most Fraction Votes')