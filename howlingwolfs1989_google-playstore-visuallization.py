import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()
%matplotlib inline
data = pd.read_csv('../input/googleplaystore.csv')
df = data.copy()
df.head()
df.info()
df.isnull().sum()
df.describe(include='all')
df['Reviews'] = df['Reviews'].str.replace('3.0M', '3000000')
df['Reviews'] = df['Reviews'].astype(np.float)
df['Price'].unique()
df['Price In Dollors'] = df['Price']
df['Price In Dollors'] = df['Price In Dollors'].str.replace('Everyone', '$0')
df['Price In Dollors'] = df['Price In Dollors'].str.replace('$', '')
df['Price In Dollors'] = df['Price In Dollors'].astype(np.float)
df.loc[10472]
df.loc[10472] = df.loc[10472].shift(periods=1, axis=0)
df['Rating'] = df['Rating'].astype(np.float64)
df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df['Installs'].unique()
df['Installs'] = df['Installs'].str.replace('+', '')
df['Installs'] = df['Installs'].str.replace(',', '')
df['Installs'] = df['Installs'].astype(np.int)
df['Type'].unique()
df['Type'].fillna(value='Free', inplace=True)
df['Content Rating'].unique()
df['Rating'].fillna(0, inplace=True)
df['Content Rating'].unique()
df['Content Rating'].fillna('Unrated', inplace=True, axis=0)
df['Current Ver'].fillna('Unknown', inplace=True, axis=0)
df['Android Ver'].fillna('Unknown', inplace=True, axis=0)
sns.set(style="ticks", color_codes=True, font_scale=1.5)
plt.figure(figsize=(5, 4))
sns.barplot(x='Type', y='Price In Dollors', ci=None, data=df);
plt.figure(figsize=(25, 8))
sns.barplot(x='Rating', y='Price In Dollors', data=df, ci=None);
df['Rating Size'] = ''
df.loc[(df['Rating']>=0.0) & (df['Rating']<=1.0), 'Rating Size'] = '0.0 - 1.0'
df.loc[(df['Rating']>=1.0) & (df['Rating']<=2.0), 'Rating Size'] = '1.0 - 2.0'
df.loc[(df['Rating']>=2.0) & (df['Rating']<=3.0), 'Rating Size'] = '2.0 - 3.0'
df.loc[(df['Rating']>=3.0) & (df['Rating']<=4.0), 'Rating Size'] = '3.0 - 4.0'
df.loc[(df['Rating']>=4.0) & (df['Rating']<=5.0), 'Rating Size'] = '4.0 - 5.0'
df.loc[(df['Rating']>=5.0) & (df['Rating']<=6.0), 'Rating Size'] = '5.0 - 6.0'
df.loc[(df['Rating']>=6.0) & (df['Rating']<=7.0), 'Rating Size'] = '6.0 - 7.0'
df.loc[(df['Rating']>=7.0) & (df['Rating']<=9.0), 'Rating Size'] = '7.0 - 8.0'
df.loc[df['Rating']>=9.0, 'Rating Size'] = '9.0+'
plt.figure(figsize=(25, 8))
sns.barplot(x='Rating Size', y='Price In Dollors', data=df, ci=None, order=['0.0 - 1.0','1.0 - 2.0','2.0 - 3.0','3.0 - 4.0','4.0 - 5.0','5.0 - 6.0','6.0 - 7.0','7.0 - 8.0','9.0+']);
plt.figure(figsize=(20, 8))
sns.lineplot(x='Rating Size', y='Price In Dollors', data=df);
plt.figure(figsize=(20, 8))
sns.relplot(x='Price In Dollors', y='Rating Size', hue='Type', data=df);
plt.figure(figsize=(15, 8))
sns.boxplot(x='Price In Dollors', y='Rating Size', hue='Type', order=['0.0 - 1.0','1.0 - 2.0','2.0 - 3.0','3.0 - 4.0','4.0 - 5.0','5.0 - 6.0','6.0 - 7.0','7.0 - 8.0','9.0+'], data=df);
rating = df.groupby('Rating Size')['Price In Dollors'].sum().reset_index()
plt.figure(figsize=(25, 8))
sns.barplot(x='Rating Size', y='Price In Dollors', data=rating, ci=None, order=['0.0 - 1.0','1.0 - 2.0','2.0 - 3.0','3.0 - 4.0','4.0 - 5.0','5.0 - 6.0','6.0 - 7.0','7.0 - 8.0','9.0+']);
plt.figure(figsize=(20, 8))
sns.lineplot(x='Rating Size', y='Price In Dollors', data=rating);
plt.figure(figsize=(20, 8))
sns.barplot(x='Content Rating', y='Price In Dollors', data=df, ci=None);
content_rating = df.groupby('Content Rating')['Price In Dollors'].sum().reset_index()
plt.figure(figsize=(20, 8))
sns.barplot(x='Content Rating', y='Price In Dollors', data=content_rating, ci=None);
plt.figure(figsize=(35, 45))
sns.barplot(x='Price In Dollors', y='Genres', data=df, ci=None);
genres = df.groupby('Genres')['Price In Dollors'].sum().reset_index()
ten_genres = genres.sort_values(by='Price In Dollors', ascending=False).reset_index(drop=True)
ten_genres.head()
plt.figure(figsize=(20, 8))
sns.barplot(x='Price In Dollors', y='Genres', data=ten_genres, ci=None, order=ten_genres.Genres.loc[:9]);
plt.figure(figsize=(20, 8))
sns.countplot(x='Rating Size', hue='Type', order=['0.0 - 1.0','1.0 - 2.0','2.0 - 3.0','3.0 - 4.0','4.0 - 5.0','5.0 - 6.0','6.0 - 7.0','7.0 - 8.0','9.0+'], data=df);
sns.catplot(x='Type', y='Price In Dollors', col='Rating Size',col_order=['0.0 - 1.0','1.0 - 2.0','2.0 - 3.0','3.0 - 4.0','4.0 - 5.0','5.0 - 6.0','6.0 - 7.0','7.0 - 8.0','9.0+'], data=df);
df['Year'] = ''
df.loc[:,'Year'] = pd.DatetimeIndex(df['Last Updated']).year
def get_yearby_info(y, plot=False):
    
    if y > 2018 and y < 2010:
        raise ValueError('Year starts from 2010 to 2018')
    if y is None:
        raise ValueError('Please enter year')
    
    year = df[df['Year'] == y]
    installs = year.groupby(['App','Year','Type', 'Size', 'Genres', 'Content Rating','Price In Dollors'])['Installs'].sum().reset_index()
    top_installs = installs.sort_values(by='Installs', ascending=False).reset_index(drop=True)
    
    if plot == False:
        return top_installs.head(10)
    else:
        plt.figure(figsize=(20, 20))
        plt.subplot(321)
        plt.xscale('log')
        sns.barplot(x='Installs', y=top_installs.App.loc[:9], data=top_installs);

        plt.subplot(322)
        plt.xscale('log')
        sns.distplot(top_installs['Installs']);

        plt.figure(figsize=(40, 10))
        plt.subplot(323)
        plt.xscale('log')
        sns.boxplot(x='Installs', y='Type', data=top_installs);
get_yearby_info(2018)
get_yearby_info(2018, plot=True)