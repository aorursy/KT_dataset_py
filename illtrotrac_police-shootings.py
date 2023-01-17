# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#Importing Librarires

import numpy as np

import pandas as pd 

import os

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline



# We dont Probably need the Gridlines. Do we? If yes comment this line

sns.set(style="ticks")



flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"] # defining the colour palette

flatui = sns.color_palette(flatui)



from wordcloud import WordCloud



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/data-police-shootings/fatal-police-shootings-data.csv") #reading data sheet
#looking at first 10 rows of dataset

df.head(10)
df.shape
#printing summary of dataframe



df.info()
#listing columns

df.columns
#modifying datetime

df['month'] = pd.to_datetime(df['date']).dt.month

df['year'] = pd.to_datetime(df['date']).dt.year
#plotting heatmap of columns

f,ax = plt.subplots(figsize=(10, 5))

sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
#Checking columns for na values to be imputed



df.columns[df.isna().any()]
#Filling missing na values with Unknown

df.fillna(0, inplace=True)

df.head(20)
#grouping stats by race

df.groupby("race").count()
#replacing race columns so they are less confusing

df.replace(to_replace = ['A'], value = ['Asian'], inplace = True)

df.replace(to_replace = ['B'], value = ['Black'], inplace = True)

df.replace(to_replace = ['H'], value = ['Hispanic'], inplace = True)

df.replace(to_replace = ['N'], value = ['Native American'], inplace = True)

df.replace(to_replace = ['O'], value = ['Other'], inplace = True)

df.replace(to_replace = ['W'], value = ['White'], inplace = True)

df.head(20)

df.groupby("gender").count()
df['manner_of_death'].value_counts()
#rough overview of age groups involved

df.groupby('age').describe()

#grouping by state

df['state'].value_counts()


deathbyshooting=df[df['manner_of_death']=='shot']

fig = px.histogram(deathbyshooting,x='race',color='race')

fig.show()
#Histogram displaying cases by age groups

fig = px.histogram(df['age'],x='age',color='age')

fig.show()
fig = px.histogram(df['state'],x='state',color='state')

fig.show()
sns.countplot(x = "manner_of_death", data = df)


fig = px.histogram(df['armed'],x='armed',color='armed')

fig.show()
#Taking a closer look at unarmed cases

unarmedcases = df.loc[df.armed == 'unarmed']

unarmedcases
fig = px.histogram(unarmedcases,x='race',color='race')

fig.show()
fig = px.histogram(unarmedcases,x='gender',color='gender')

fig.show()
fig = px.histogram(unarmedcases,x='flee',color='flee')

fig.show()
import plotly.figure_factory as ff

np.random.seed(1)

x = df['age']

hist_data = [x]

group_labels = ['Age']

fig = ff.create_distplot(hist_data, group_labels)

fig.show()
mentallyill = df[df.signs_of_mental_illness==True]

mentallyill = mentallyill[['year','race']]

mentallyill['kills'] = 1

mentallyill = mentallyill.groupby(['year','race']).sum()

mentallyill = mentallyill.reset_index()

fig = px.bar(mentallyill, y='kills', x='year',color='race', barmode='group')

fig.show()