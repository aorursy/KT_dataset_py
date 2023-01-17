# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/joox.csv")

df.drop(['Unnamed: 0'], axis=1, inplace=True)

df.drop(['Track.Name'], axis=1, inplace=True)

print(df)
artist = df['Artist.Name'].nunique()

print(artist)
var = df['Artist.Name'].value_counts().reset_index().rename(columns = {'index' : 'Artist Name'})

top = pd.DataFrame(var)

top.rename(columns={'Artist.Name': 'Total'}, inplace=True)

plt.style.use('seaborn')

plt.figure(figsize=(15,8))





p = sns.barplot(x=top['Total'], y=top['Artist Name'], linewidth=0.5)

p.axes.set_title("Total Number of Top 50 Songs by Artist",fontsize=20)

p.set_xlabel("Number of Songs",fontsize=15)

p.set_ylabel("Artist Name",fontsize=15)

plt.show()
plt.style.use('seaborn')

plt.figure(figsize=(15,8))

p = sns.countplot(y='Genre', data = df.reset_index(),order = df['Genre'].value_counts().index, linewidth=0.5)

p.axes.set_title("Total Number of Songs by Genre",fontsize=20)

p.set_xlabel("Number of Songs",fontsize=15)

p.set_ylabel("Genre",fontsize=15)

plt.show()
plt.style.use('seaborn')

plt.figure(figsize=(15,8))

p = sns.regplot(x="Speechiness.", y="Popularity", data=df)

p.axes.set_title("Popularity by Speechiness. of the song",fontsize=20)

p.set_xlabel("Speechiness.",fontsize=15)

p.set_ylabel("Popularity",fontsize=15)

p.tick_params(labelsize=15)