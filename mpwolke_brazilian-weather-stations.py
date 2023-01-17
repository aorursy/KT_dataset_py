# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import unicodedata

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/automatic-weather-stations-brazil/automatic_stations_codes.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'automatic_stations_codes.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
g = sns.lmplot(x="LONGITUDE", y="LATITUDE", data=df,

           fit_reg=False, scatter_kws={"s": 30}, hue='REGIAO', height=10)

plt.title('Brazilian Weather Stations')

plt.show()
plt.figure(figsize=(20,12))

g = sns.scatterplot(x='LONGITUDE', y='LATITUDE', data=df, hue='UF')

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1);
fig_px = px.scatter_mapbox(df, lat="LATITUDE", lon="LONGITUDE",

                           hover_name="REGIAO",

                           zoom=11, height=300)

fig_px.update_layout(mapbox_style="open-street-map",

                     margin={"r":0,"t":0,"l":0,"b":0})



fig_px.show()
fig_px.update_traces(marker={"size": [10 for x in df]})
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
plot_count("ALTITUDE", "ALTITUDE", df,4)
fig = px.pie(df, values=df['ALTITUDE'], names=df['UF'],

             title='Brazilian Weather Stations',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
sns.countplot(x="UF",data=df,palette="GnBu_d",edgecolor="black")

plt.title('Brazilian Weather Stations', weight='bold')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)