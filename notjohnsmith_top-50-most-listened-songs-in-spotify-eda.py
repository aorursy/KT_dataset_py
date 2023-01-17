import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style='white')
df = pd.read_csv('../input/top50spotify2019/top50.csv',encoding='latin-1')

df.rename(columns={'Unnamed: 0':'Rank'}, inplace=True)

df.head()
plt.figure(figsize=(10,20))

sns.countplot(y='Artist.Name',data=df,order=df['Artist.Name'].value_counts().index).set(title='Number of Songs per Artist',

                                                                                       xlabel='Number of Songs',

                                                                                       ylabel='Artist')
plt.figure(figsize=(10,12))

sns.countplot(y='Genre',data=df,order=df['Genre'].value_counts().index).set(title='Number of Songs per Genre',

                                                                                       xlabel='Number of Songs',

                                                                                       ylabel='Genre')
plt.figure(figsize=(15,5))

sns.distplot(df['Beats.Per.Minute']).set(title='Beats per Minute Distribution',

                                        xlabel='Beats per Minute',

                                        ylabel='Number of Songs (Normalized)')
print('BPM:')

print("Median:",df['Beats.Per.Minute'].median())

print("Standard Deviation:",df['Beats.Per.Minute'].std())

df['Beats.Per.Minute'].describe()
plt.figure(figsize=(10,15))

sns.countplot(y=df['Beats.Per.Minute'],order=df['Beats.Per.Minute'].value_counts().index).set(title='Beats per Minute Distribution',

                                        xlabel='Beats per Minute',

                                        ylabel='Number of Songs (Normalized)')
plt.figure(figsize=(15,5))

sns.distplot(df['Energy']).set(title='Energy Distribution',

                                        xlabel='Energy',

                                        ylabel='Number of Songs (Normalized)')
print('Energy:')

print("Median:",df['Energy'].median())

print("Standard Deviation:",df['Energy'].std())

df['Energy'].describe()
plt.figure(figsize=(15,5))

sns.distplot(df['Danceability']).set(title='Danceability Distribution',

                                        xlabel='Danceability',

                                        ylabel='Number of Songs (Normalized)')
temp_param = 'Danceability'

print(temp_param)

print("Median:",df[temp_param].median())

print("Standard Deviation:",df[temp_param].std())

df[temp_param].describe()
plt.figure(figsize=(15,5))

sns.distplot(df['Loudness..dB..']).set(title='Loudness Distribution',

                                        xlabel='Loudness',

                                        ylabel='Number of Songs (Normalized)')
temp_param = 'Loudness..dB..'

print(temp_param)

print("Median:",df[temp_param].median())

print("Standard Deviation:",df[temp_param].std())

df[temp_param].describe()
plt.figure(figsize=(15,5))

sns.distplot(df['Liveness']).set(title='Liveness Distribution',

                                        xlabel='Liveness',

                                        ylabel='Number of Songs (Normalized)')
plt.figure(figsize=(15,5))

sns.distplot(df['Length.']).set(title='Length Distribution',

                                        xlabel='Length (Minutes)',

                                        ylabel='Number of Songs (Normalized)')
temp_param = 'Length.'

print(temp_param)

print("Median:",df[temp_param].median())

print("Standard Deviation:",df[temp_param].std())

df[temp_param].describe()
plt.figure(figsize=(15,5))

sns.distplot(df['Valence.']).set(title='Valence Distribution',

                                        xlabel='Valence',

                                        ylabel='Number of Songs (Normalized)')
temp_param = 'Valence.'

print(temp_param)

print("Median:",df[temp_param].median())

print("Standard Deviation:",df[temp_param].std())

df[temp_param].describe()
plt.figure(figsize=(15,5))

sns.distplot(df['Acousticness..'],kde=False).set(title='Acousticness Distribution',

                                        xlabel='Acousticness',

                                        ylabel='Number of Songs (Normalized)')
temp_param = 'Acousticness..'

print(temp_param)

print("Median:",df[temp_param].median())

print("Standard Deviation:",df[temp_param].std())

df[temp_param].describe()
plt.figure(figsize=(15,5))

sns.distplot(df['Speechiness.'],kde=False).set(title='Speechiness Distribution',

                                        xlabel='Speechiness',

                                        ylabel='Number of Songs (Normalized)')
temp_param = 'Speechiness.'

print(temp_param)

print("Median:",df[temp_param].median())

print("Standard Deviation:",df[temp_param].std())

df[temp_param].describe()
plt.figure(figsize=(15,5))

sns.distplot(df['Popularity']).set(title='Popularity Distribution',

                                        xlabel='Popularity',

                                        ylabel='Number of Songs (Normalized)')
temp_param = 'Popularity'

print(temp_param)

print("Median:",df[temp_param].median())

print("Standard Deviation:",df[temp_param].std())

df[temp_param].describe()
rank_corr = df.corr()['Rank'].drop(index='Rank').sort_values(ascending=False)

print(rank_corr)
sns.pairplot(df,

             x_vars='Rank',

             y_vars=df[rank_corr.index].columns,

             kind='reg',

             aspect=7)