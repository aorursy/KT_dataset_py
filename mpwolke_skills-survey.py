# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsskillscsv/skills.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'skills.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
df.isnull().sum()
df.columns.tolist()
#Code from Gabriel Preda

#plt.style.use('dark_background')

def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set2')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("Financial services", "Financial services", df,4)
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'England', data = df, palette="PuRd",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(132)

sns.countplot(x= 'Scotland', data = df, palette="ocean",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= 'Wales', data = df, palette="Set3",edgecolor="black")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'Low', data = df, palette="cividis",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(132)

sns.countplot(x= 'Medium', data = df, palette="PuRd",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= 'High', data = df, palette="viridis",edgecolor="black")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'Charity / voluntary sector', data = df, palette="cubehelix",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(132)

sns.countplot(x= 'Education', data = df, palette="flag",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= 'Construction', data = df, palette="bone",edgecolor="black")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'Public admin.', data = df, palette="coolwarm",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(132)

sns.countplot(x= 'Financial services', data = df, palette="afmhot",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= 'Business services', data = df, palette="rainbow",edgecolor="black")

plt.xticks(rotation=45)

plt.show()
labels = df['Charity / voluntary sector'].value_counts().index

size = df['Charity / voluntary sector'].value_counts()

colors=['#BFBF3F','#44BF3F']

plt.pie(size, labels = labels, colors = colors, shadow = True, autopct='%1.1f%%',startangle = 90)

plt.title('Charity / voluntary sector Skills', fontsize = 20)

plt.legend()

plt.show()
fig = px.pie(df, values=df['England'], names=df['Medium'],

             title='Medium Skills',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=df['Yes'][0:10],

    y=df['No'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Skills Survey',

    xaxis_title="Yes",

    yaxis_title="No",

)

fig.show()
fig = go.Figure(data=[go.Bar(

            x=df['Yes'][0:10], y=df['No'][0:10],

            text=df['No'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Skills Survey',

    xaxis_title="Yes",

    yaxis_title="No",

)

fig.show()