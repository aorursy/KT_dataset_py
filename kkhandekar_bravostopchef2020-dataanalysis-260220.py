#Importing Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #data visualisation

import matplotlib.pyplot as plt #data visualisation

%matplotlib inline

#Data Load

data = pd.read_csv('../input/bravos-top-chef-2020-all-stars-past-results/Top Chef Allstar Contestants Previous Wins.csv',header='infer')
data.shape
#Checking for missing values

data.isna().sum()
data.head()
#Season 12 - Data Visualisation

sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(15,10))

ax =sns.swarmplot (x='Placement', y='Week num', data=data[data['Prev Season Num']==12], hue = 'Contestant', palette="GnBu_d")

sns.despine(trim=True, left=True)

plt.title('Season 12 - Data Visualisation')

plt.ylabel('Week')

plt.xlabel('Placement')
#Season 6 - Data Visualisation

sns.set(style="darkgrid")

fig = plt.figure()

g = sns.catplot(x="Quick Fire Win", y="Week num", hue="Contestant", data=data[data['Prev Season Num']==6],

                height=10, kind="swarm", palette="muted")

#fig.fig.set_size_inches(15,10)

plt.title("Season 6 - Data Visualisation")



plt.show()
#Season 16 - Data Visualisation

sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(15,10))

ax =sns.swarmplot (x='Week num', y='Placement', data=data[data['Prev Season Num']==16], hue = 'Contestant', palette="GnBu_d")

sns.despine(trim=True, left=True)

plt.title('Season 16 - Data Visualisation')

plt.ylabel('Placement')

plt.xlabel('Weeks')
#Season 16 - Data Visualisation

sns.set(style="darkgrid")

fig = plt.figure()

g = sns.catplot(x="Contestant", y="Week num", hue="Placement", data=data[data['Prev Season Num']==16],

                height=10, kind="swarm", palette="muted")

#fig.fig.set_size_inches(15,10)

plt.title("Season 16 - Data Visualisation")



plt.show()
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(10, 15))



sns.set_color_codes("pastel")

sns.barplot(x="Prev Season Num", y="Contestant", data=data,

            label="Prev. Season", color="b")



sns.set_color_codes("muted")

sns.barplot(x="Week num", y="Contestant", data=data,

            label="Weeks", color="b")



ax.legend(ncol=2, loc="upper right", frameon=False)

ax.set(xlim=(0, 24), ylabel=" ",

       xlabel="Seasons / Weeks")

sns.despine(left=True, bottom=True)