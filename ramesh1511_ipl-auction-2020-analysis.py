import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ipl_data = pd.read_csv("../input/ipl-auction-dataset/IPL Data.csv",engine = 'python')
ipl_data.head()
ipl_data.describe()
ipl_data.shape
ipl_data.info()
ipl_data.drop(['S.No','Country'], axis=1, inplace=True)
ipl_data.head()
cap_count  = ipl_data['Capped / Uncapped /Associate'].value_counts()

ax = cap_count.plot(kind='bar', color=['c','y'])

ax.set_ylabel('number of players')

ax.set_title('capped and uncapped players')

x_offset = -0.05

y_offset = 0.02

for p in ax.patches:

    b = p.get_bbox()

    val = "{:.0f}".format(b.y1 + b.y0)        

    ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
pd.DataFrame(ipl_data.groupby(["Playing Role", "Capped / Uncapped /Associate"]).size().rename('Freq'))
# plot data

# fig, ax = plt.subplots(figsize=(15,7))

# fig.tight_layout()

# use unstack()

ipl_data.groupby(["Playing Role", "Capped / Uncapped /Associate"]).size().rename('Freq').unstack().plot(kind='pie',figsize=(15,7),subplots=True)



# type_show_ids = plt.pie(rating_counts, labels=rating_labels, autopct='%1.1f%%', shadow=True, colors=colors)

plt.show()
df = ipl_data[ipl_data['IPL Matches']==ipl_data['IPL Matches'].max()]

print("Player who has played highest IPL matches")

df[['Name', 'Playing Role', 'IPL Matches']].style.hide_index()
print("Number of players making Debut in IPL")

debut_df = ipl_data[ipl_data['IPL Team(s)'].isnull() & ipl_data['IPL 2019 Team'].isnull()]

debut_df.Name.count()
print("All the players making Debut in IPL")

debut_df[['Name']].style.hide_index()
print("Number of players who played IPL before 2019 and re-selected in 2020")

df_20 = ipl_data[ipl_data['IPL Team(s)'].notnull() & ipl_data['IPL 2019 Team'].isnull()]

df_20.Name.count()
print("All the players making Debut in IPL")

df_20[['Name']].style.hide_index()
df = ipl_data[ipl_data['Auctioned Price(in ₹ Lacs)']==ipl_data['Auctioned Price(in ₹ Lacs)'].max()]

print("Highest Paid player")

df[['Name', 'Playing Role','IPL 2020 Team', 'Auctioned Price(in ₹ Lacs)']].style.hide_index()
uncapped_df = ipl_data[ipl_data['Capped / Uncapped /Associate'] == 'Uncapped']

uncapped_df = uncapped_df[uncapped_df['Auctioned Price(in ₹ Lacs)']==uncapped_df['Auctioned Price(in ₹ Lacs)'].max()]

print("Highest uncapped player")

uncapped_df[['Name', 'Playing Role','IPL 2020 Team', 'Auctioned Price(in ₹ Lacs)']].style.hide_index()
role_count = ipl_data['Playing Role'].value_counts()

ax = role_count.plot(kind='bar')

ax.set_ylabel('number of players')

ax.set_title('Players roles distribution')

x_offset = -0.05

y_offset = 0.02

for p in ax.patches:

    b = p.get_bbox()

    val = "{:.0f}".format(b.y1 + b.y0)        

    ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
ipl_data.head()
ipl_data["IPL Team(s)"].fillna(" ", inplace = True)

ipl_data['Total Teams played'] = [len(x.split(',')) for x in ipl_data['IPL Team(s)'].tolist()]
# selecting only non-debut players for 2020 IPL

single_match_df = ipl_data[~ipl_data.isin(debut_df)].dropna()

single_match_df.head()
print("players who had played for only one franchise before IPL 2020")

one_df = single_match_df[single_match_df['Total Teams played'] == 1]

one_df[['Name', 'Playing Role','IPL 2020 Team']].style.hide_index()