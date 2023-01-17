import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

%matplotlib inline

sns.set_style('white')
df = pd.read_csv('../input/rolling-stones-top-500-songs-of-all-time/rollingstone.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)
top_10_artists = df['Artist'].value_counts().head(10)

plt.barh(top_10_artists.index, top_10_artists)

plt.show()
df['Ranking'] = np.arange(499,-1, -1)

a = df['Popularity'].value_counts()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,5))



fig.suptitle('COMPARING QUALITY AND POPULARITY', fontdict={'fontsize':24})



ax1.scatter(df['Ranking'], df['Popularity'])

ax1.set_title('Popularity vs. ranking')

ax1.set_xlabel('Popularity')

ax1.set_ylabel('Rank')

ax2.hist(df['Popularity'])

ax2.set_title('Popularity Distribution')

plt.show()
df.sort_values(by='Popularity', axis =0, ascending=False).head(10).iloc[:, np.r_[:2 ,6]]
df.loc[:, ['Artist', 'Title', 'Popularity']].tail(10)
# create a dataframe with just the features so we can iterate over it

features = df.copy().iloc[:, np.r_[7:18,-3:-1]]



# convert duration column from milliseconds to seconds

features['duration_sec'] = features['duration_ms'] / 1000 / 60 

features['duration_sec'] = round(features['duration_sec'], 2)

features2 = features.drop('duration_ms', axis=1)



# setup subplots

rows = 4

cols = 4

fig, ax = plt.subplots(rows, cols, figsize=(20,15)) #  

titles = features2.columns.values

fig.suptitle('DISTRIBUTION OF AUDIO FEATURES', fontweight= 'bold')

# counter to choose which feature to plot

title_no = 0 



# given that we have odd size let's deactivate plots that go over array size

while title_no < titles.size:

        for row in range(rows):

            for col in range(cols):

                if title_no >= titles.size:

                    ax[row,col].set_visible(False)

                    sns.distplot(features2[titles[0]])

                else:

                    sns.distplot(features2[titles[title_no]], kde=False, ax=ax[row,col])

                    sns.despine()

                    title_no +=1
rows = 4

cols = 4

fig, ax = plt.subplots(rows, cols, figsize=(20,15)) #  

titles = features2.columns.values

fig.suptitle('VIOLINPLOT DISTRIBUTION OF AUDIO FEATURES', fontweight= 'bold')

# counter to choose which feature to plot

title_no = 0 



# given that we have odd size let's deactivate plots that go over array size

while title_no < titles.size:

        for row in range(rows):

            for col in range(cols):

                if title_no >= titles.size:

                    ax[row,col].set_visible(False)

                    sns.distplot(features2[titles[0]])

                else:

                    sns.violinplot(features2[titles[title_no]], ax=ax[row,col])

                    sns.despine()

                    title_no +=1
data = pd.concat([features2, df['Ranking']], axis=1)

sns.heatmap(data.corr(), cmap='coolwarm')

plt.show()