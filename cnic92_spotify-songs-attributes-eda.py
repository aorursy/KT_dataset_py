import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Load data

df = pd.read_csv('../input/spotify-past-decades-songs-50s10s/1950.csv')



# Show info and description of dataset

print(df.head(5))

df.info()

df.describe()
# Set column Number as index

df.set_index('Number', inplace=True)



# Drop year column

df.drop(['year'], axis=1, inplace=True)



# Show updated dataframe

df.head(5)
# Find missing values

df.isna().sum()
df.dropna(how='any', inplace=True)



# Take a look at the cleaned dataset

print(df.head(5))

df.info()

df.describe()
# Find percent of each genre

df_genre = df['top genre'].value_counts() / len(df)

sizes = df_genre.values.tolist()

labels = df_genre.index.values.tolist()



# Pie chart for genre

fig1, ax1 = plt.subplots(figsize=(10,10))

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, textprops={'fontsize': 14})

ax1.axis('equal')

plt.show()
# Plot boxplot (variables only)

sns.boxplot(data=df.drop(['title', 'artist', 'top genre'], axis=1))

plt.xlabel('Features')

plt.ylabel('Value')

plt.show()
sns.pairplot(data=df.drop(['title', 'artist'], axis=1), hue='top genre')

plt.show()
# Top 3 most popular genres

genre_list = ['adult standards', 'brill building pop', 'deep adult standards']



# Extract sample from df

df_ = df.loc[df['top genre'].isin(genre_list)]



# Plot pairplot with limited data

sns.pairplot(data=df_.drop(['title', 'artist'], axis=1), hue='top genre')

plt.show()
# Plot linear correlation matrix

fig, ax = plt.subplots(figsize=(15,10))

sns.heatmap(df.corr(), annot=True, cmap='YlGnBu', vmin=-1, vmax=1, center=0, ax=ax)

plt.title('LINEAR CORRELATION MATRIX')

plt.show()