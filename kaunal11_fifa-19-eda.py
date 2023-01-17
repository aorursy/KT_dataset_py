# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import chart_studio.plotly  as py

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/fifa19/data.csv')

replace_values = {'Messi' : 'Normal', 'C. Ronaldo' : 'Lean', 'Neymar' : 'Lean', 'PLAYER_BODY_TYPE_25':'Normal', 'Shaqiri': ' Normal', 'Akinfenwa': 'Normal' }  

df = df.replace({"Body Type": replace_values})  
# Function to create pie charts by Categorical variable



def pie_count(data,field, title='Players by '):

    title+=field

    if data[field].isnull().sum():

        data = data[data[field].notna()]

    labels=list(data[field].unique())[:11]

    sizes=list(data[field].value_counts())[:11]

    

    # Plot

    plt.pie(sizes, labels=labels,autopct='%1.1f%%', shadow=True, startangle=140)

    patches, texts = plt.pie(sizes, shadow=True, startangle=90)

    plt.title(title)

    plt.legend(patches, labels, loc="best")

    plt.axis('equal')

    plt.tight_layout()

    plt.show()
pie_count(df, 'Nationality')

pie_count(df, 'Club')

pie_count(df, 'Preferred Foot')

pie_count(df, 'Body Type')

pie_count(df, 'Position')
df['Value']=df['Value'].str.replace('€','')

df['Value']=df['Value'].str.replace('M',' 1000000')

df['Value']=df['Value'].str.replace('K',' 1000')
df['Value'] = df['Value'].str.split(' ', expand=True)[0].astype(float) * df['Value'].str.split(' ', expand=True)[1].astype(float)
# Top 20 players by value



df.sort_values(by='Value',ascending=False)[['Name','Value']].head(20)
# Top 20 players by value - Graphical represenation



df.sort_values(by='Value',ascending=False)[['Name','Value']].head(20).plot.bar(x='Name',y='Value')
# Mean Value of a player by age



df.groupby('Age')['Value'].mean().head(20).plot.bar(x='Age',y='Value')
# Top 20 teams by Value



df.groupby('Club')['Value'].mean().reset_index().sort_values('Value', ascending=False).head(20)
# Top 20 teams by Value - Graphical representation



df.groupby('Club')['Value'].mean().reset_index().sort_values('Value', ascending=False).head(20).plot.bar(x='Club',y='Value')
# Top 20 teams by Average player Rating



df.groupby('Club')['Overall'].mean().reset_index().sort_values('Overall', ascending=False).head(20)
# Top 20 teams by Average player Rating - Graphical Representation



df.groupby('Club')['Overall'].mean().reset_index().sort_values('Overall', ascending=False).head(20).plot.bar(x='Club',y='Overall')