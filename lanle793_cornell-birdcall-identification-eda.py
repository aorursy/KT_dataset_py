# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



"""

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

"""



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go



# Import data

train_csv = pd.read_csv("../input/birdsong-recognition/train.csv")

train_csv.info()
print("There are {:,} unique bird species in the dataset.".format(len(train_csv['species'].unique())))
print("There are {:,} audio files in the dataset.".format(len(train_csv['filename'].unique())))
# Plot the counts of first 20 bird species sorted by quantity

train_csv['species'].value_counts().head(20).plot.bar()
# Plot the counts of first 20 bird species sorted alphabetically

train_csv['species'].value_counts().sort_index(ascending=True).head(20).plot.bar()
top_10 = list(train_csv['elevation'].value_counts().head(10).reset_index()['index'])

data = train_csv[train_csv['elevation'].isin(top_10)]



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['elevation'], palette="hls", order = data['elevation'].value_counts().index)



plt.title("Top 10 Elevation Types", fontsize=16)
top_10 = list(train_csv['country'].value_counts().head(10).reset_index()['index'])

data = train_csv[train_csv['country'].isin(top_10)]



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['country'], palette="hls", order = data['country'].value_counts().index)



plt.title("Top 10 Countries with bird recordings", fontsize=16)
def get_year(date):

    return date.split('-')[0]



train_csv['year'] = train_csv['date'].apply(get_year)



top_25 = list(train_csv['year'].value_counts().head(25).reset_index()['index'])

data = train_csv[train_csv['year'].isin(top_25)]



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['year'], palette="hls")
train_csv['bird_seen'].fillna('Not Defined',inplace=True)

labels = train_csv['bird_seen'].value_counts().index

values = train_csv['bird_seen'].value_counts().values

colors=['#3795bf','#bfbfbf', '#cf5353']



fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial',marker=dict(colors=colors))])

fig.show()
plt.figure(figsize=(16, 6))

ax = sns.countplot(data['number_of_notes'], palette="hls", order = data['number_of_notes'].value_counts().index)

plt.xlabel("Number of notes", fontsize=14)
train_csv['playback_used'].fillna('Not Defined',inplace=True)

labels = train_csv['playback_used'].value_counts().index

values = train_csv['playback_used'].value_counts().values

colors=['#3795bf','#bfbfbf', '#cf5353']



fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial',marker=dict(colors=colors))])

fig.show()
print(train_csv['duration'].describe())
train_csv['channels'].fillna('Not Defined',inplace=True)

labels = train_csv['channels'].value_counts().index

values = train_csv['channels'].value_counts().values

colors=['#3795bf','#bfbfbf', '#cf5353']



fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial',marker=dict(colors=colors))])

fig.show()