# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime



#Importing Data plotting libraries

import matplotlib.pyplot as plt     

import seaborn as sns      

import plotly.express as px

import plotly.graph_objects as go



# Text Analytics

from wordcloud import WordCloud



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
killings = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding="windows-1252")

percent_over25 = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv", encoding="windows-1252")

median_household_income = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv", encoding="windows-1252")

below_poverty = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv", encoding="windows-1252")
killings['date'] = killings['date'].apply(lambda x: datetime.strptime(x, "%d/%m/%y"))

killings['year_month'] = killings['date'].apply(lambda x: x.strftime('%m/%Y'))



killings.head()
state_wise = killings.groupby('state')['id'].agg('count').reset_index()

state_wise = state_wise.sort_values('id', ascending=False)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=state_wise['state'], y=state_wise['id'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('Killings count')

plt.title("Statewise police killings in the US")
race = killings.groupby('race')['id'].agg('count').reset_index()

race = race.sort_values('id', ascending=False)

labels = ['White', 'Black', 'Hispanic', 'Asian', 'Native American', 'Others']



#Vizualization

fig = go.Figure()

fig.add_trace(go.Bar(

    x=labels,

    y=race['id']

))

fig.update_layout(

    title="Most killed Races",

    xaxis_title="Races",

    yaxis_title="Killings Count",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f")

)

fig.show()
racial_data = killings[(killings['race'] == 'W') | (killings['race'] == 'B') | (killings['race'] == 'H')]
pd.options.mode.chained_assignment = None  # default='warn'

# Let's assign age groups

bins = [0, 10, 20, 30, 40, 50, 60, 70, np.inf]

labels = ['0', '10s', '20s', '30s', '40s', '50s', '60s', '70+']

racial_data['age_groups'] = pd.cut(killings['age'], bins, labels=labels)

age_data = racial_data.groupby(['age_groups', 'race'])['id'].agg('count').reset_index()



fig = px.line(age_data, x='age_groups', y='id', color='race')

fig.update_layout(

    title="Agewise killings among most killed races",

    xaxis_title="Age Groups",

    yaxis_title="Killings Count",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f")

)

fig.show()

arms = racial_data['armed'].values

wordcloud = WordCloud(background_color = 'white', min_font_size = 3, max_words=20).generate(str(arms))

plt.figure(figsize = (8, 8), facecolor = None)

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
fleeing = racial_data.groupby(['flee', 'race'])['id'].agg('count').reset_index()



#Let's visualize

fig = px.bar(fleeing, x='flee', y='id', color='race')

fig.update_layout(

    title="Were the people trying to flee at the time of death",

    xaxis_title="Flee",

    yaxis_title="People Count",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f")

)

fig.show()