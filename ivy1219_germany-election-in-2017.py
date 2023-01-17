

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

df= pd.read_csv('/kaggle/input/german-election-2017/2017_german_election_overall.csv')

party = pd.read_csv('/kaggle/input/german-election-2017/2017_german_election_party.csv')

print(df.shape,party.shape)
df['vote_on_voter']=df['total_votes']/df['registered.voters']

plt.figure(figsize=(10,16))

sns.boxenplot(y='state',x = 'vote_on_voter',data= df,palette='Set1')

plt.axvline(x =0.76,linewidth =2.5, linestyle ='--',color = 'grey')

plt.text(0.77,1,'The median rate:0.76',color ='black',fontsize = 14)
# find the top5 supported parties

party.groupby(['party'])['votes_first_vote'].sum().sort_values(ascending =False)[:5]
party[party['party'] == 'Freie.Demokratische.Partei']['votes_first_vote'].describe()
tmp=party.loc[(party['party']=='Christlich.Demokratische.Union.Deutschlands') |(party['party']=='Sozialdemokratische.Partei.Deutschlands')

             |(party['party']=='Alternative.für.Deutschland')|(party['party']=='DIE.LINKE')|(party['party']=='BÜNDNIS.90.DIE.GRÜNEN') | (party['party']=='Freie.Demokratische.Partei') ]



g = sns.FacetGrid(tmp, col="state", col_wrap=4, height=3)

g.map(sns.barplot, "votes_first_vote", "party",  palette ='GnBu_d', ci=None);
#load and read geo data

import geopandas as gpd

shapefile = gpd.read_file("/kaggle/input/german-election-2017/Geometrie_Wahlkreise_19DBT_VG250_geo.shp")

shapefile.rename(columns = {'LAND_NAME':'state'},inplace =True)

#shapefile
merged = shapefile.set_index('state').join(tmp.set_index('state'))

merged = merged.reset_index()

merged.head(5)
fig, ax = plt.subplots(1, figsize=(16, 15))

ax.axis('off')

ax.set_title(' Distribution of CDU supporters ', fontdict={'fontsize': '20', 'fontweight' : '8'})



color = 'Greens'

vmin, vmax = 10000, 100000

sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)

cbar.ax.tick_params(labelsize=12)

CDU = merged[merged['party'] =='Christlich.Demokratische.Union.Deutschlands']



CDU.plot('votes_first_vote', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(16,15))

fig, ax = plt.subplots(1, figsize=(16, 16))

ax.axis('off')

ax.set_title('Distribution of AfD supporters', fontdict={'fontsize': '20', 'fontweight' : '8'})



color = 'Reds'

vmin, vmax = 10000, 100000

sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)

cbar.ax.tick_params(labelsize=12)

AFD= merged[merged['party'] =='Alternative.für.Deutschland']



CDU.plot('votes_first_vote', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(16,16))
df= df[df['vote_on_voter']>0]

merged_df = shapefile.set_index('state').join(df.set_index('state'))

#merged_df['vote_on_voter'] =df['vote_on_voter']

merged_df = merged_df.reset_index()

merged_df.columns
# just curious about FDP

fig, ax = plt.subplots(1, figsize=(16, 16))

ax.axis('off')

ax.set_title('Distribution of FDP supporters', fontdict={'fontsize': '20', 'fontweight' : '8'})



color = 'Purples'

vmin, vmax = 3000, 30000

sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)

cbar.ax.tick_params(labelsize=12)

FDP= merged[merged['party'] =='Freie.Demokratische.Partei']



FDP.plot('votes_first_vote', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(16,16))
fig, ax = plt.subplots(1, figsize=(16, 16))

ax.axis('off')

ax.set_title('voter popularity', fontdict={'fontsize': '20', 'fontweight' : '8'})



color = 'Blues'

vmin, vmax = 150000, 260000

sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)

cbar.ax.tick_params(labelsize=12)



merged_df.plot('registered.voters', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(16,16))
merged_1 = shapefile.set_index('state').join(party.set_index('state'))

merged_1 = merged_1.reset_index()

merged.head(5)
Bayern = merged_1[merged_1['state']== 'Bayern']

Bayern.groupby(['party'])['votes_first_vote'].sum().sort_values(ascending = False)[:10]
Bayern_CSU = Bayern[Bayern.party == 'Christlich.Soziale.Union.in.Bayern.e.V.']

Bayern_SPD = Bayern[Bayern.party == 'Sozialdemokratische.Partei.Deutschlands']

Bayern_AFD = Bayern[Bayern.party == 'Alternative.für.Deutschland']

plt.figure(figsize=(10,16))

sns.barplot(y='area_name',x = 'votes_first_vote',data= Bayern_CSU,color = 'firebrick',label = 'Bayern_CSU')

sns.barplot(y='area_name',x = 'votes_first_vote',data= Bayern_SPD,color = 'yellow',label = 'Bayern_SPD')

sns.barplot(y='area_name',x = 'votes_first_vote',data= Bayern_AFD,color = 'blue',alpha = 0.8,label = 'Bayern_ARD')

plt.legend()
