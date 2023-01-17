#import and preview the data

import pandas as pd

df = pd.read_csv('../input/lyrics.csv')

df.head()
df.info()
#replace carriage returns

df = df.replace({'\n': ' '}, regex=True)

df.head()
#count the words in each song

df['word_count'] = df['lyrics'].str.split().str.len()

df.head()
#check the word counts by genre

df['word_count'].groupby(df['genre']).describe()
#let's see what the songs with 1 word look like

df1 = df.loc[df['word_count'] == 1]

df1
#elimintate the 1-word songs and review the data again

df = df[df['word_count'] != 1]

df['word_count'].groupby(df['genre']).describe()
#There are still some outliers on the low end. Reviewing songs with less than 100 words.

df100 = df.loc[df['word_count'] <= 100]

df100
#let's check on the high end

df1000 = df.loc[df['word_count'] >= 1000]

df1000
#let's get rid of the outliers on the low and high end using somewhat randomly selected points

del df1, df100, df1000 

df_clean = df[df['word_count'] >= 100]

df_clean = df[df['word_count'] <= 1000]

df_clean['word_count'].groupby(df_clean['genre']).describe()
#let's see how much smaller the data set is now

df.info()
#check the overall distribution of the cleaned dataset

import seaborn as sns

sns.violinplot(x=df_clean["word_count"])
#compare wordcounts by genre

import matplotlib as mpl

mpl.rc("figure", figsize=(12, 6))

sns.boxplot(x="genre", y="word_count", data=df_clean)
genre = df_clean.groupby(['genre'],as_index=False).count()

genre2 = genre[['genre','song']]

genre2
liquor = pd.DataFrame(df_clean.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('liquor')].count()))

liquor.reset_index(inplace=True)

liquor.columns = ['genre', 'liquor_lyrics']

liquor
beer = pd.DataFrame(df_clean.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('beer')].count()))

beer.reset_index(inplace=True)

beer.columns = ['genre', 'beer_lyrics']

beer
wine = pd.DataFrame(df_clean.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('wine')].count()))

wine.reset_index(inplace=True)

wine.columns = ['genre', 'wine_lyrics']

wine
pills = pd.DataFrame(df_clean.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('pills')].count()))

pills.reset_index(inplace=True)

pills.columns = ['genre', 'pills_lyrics']

pills
weed = pd.DataFrame(df_clean.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('weed')].count()))

weed.reset_index(inplace=True)

weed.columns = ['genre', 'weed_lyrics']

weed
import functools

dfs = [genre2,beer,wine,liquor,pills,weed]

genre3 = functools.reduce(lambda left,right: pd.merge(left,right,on='genre', how='outer'), dfs)

genre3
genre3['beer_ratio'] = genre3['beer_lyrics'] / genre3['song']

genre3['wine_ratio'] = genre3['wine_lyrics'] / genre3['song']

genre3['liquor_ratio'] = genre3['liquor_lyrics'] / genre3['song']

genre3['pills_ratio'] = genre3['pills_lyrics'] / genre3['song']

genre3['weed_ratio'] = genre3['weed_lyrics'] / genre3['song']

genre3