import pandas as pd

oscars = pd.read_csv('../input/the-oscar-award/the_oscar_award.csv')

movies = pd.read_csv('../input/imdb-movie-19722019/imdb_1972-2019.csv')

oscars.info() 
import numpy as np

oscars['win_']= np.where(oscars['win']==True,1,0) #convert to numeric

oscars['decade_award'] = (10 * (oscars['year_ceremony'] // 10)).astype(str) + 's' #get the decade

oscars.head()
melted_oscars = pd.pivot_table(oscars,values='win_',index=['year_film','film','decade_award'],

                               columns='category').reset_index()

melted_oscars.replace(1,2,inplace=True)

melted_oscars.replace(0,1,inplace=True)

melted_oscars.fillna(0,inplace=True)

melted_oscars
movies.head()
df = pd.merge(movies,melted_oscars,left_on=['Title','Year'],right_on=['film','year_film'])

df.head()
df['oscar_buzz'] = df.iloc[:, 13:].sum(axis=1)

df.head()
df['oscar_buzz'].corr(df['Revenue (Millions)'])
df['Runtime (Minutes)'].corr(df['Revenue (Millions)'])
df['oscar_buzz'].corr(df['Runtime (Minutes)'])
import seaborn as sns

sns.set(font_scale=2)

plot = sns.relplot(x='Revenue (Millions)', y='oscar_buzz',hue='decade_award',kind='scatter',data=df, height=10,

           s=100, label='large')

plot.savefig("OscarsOutliers.png")
df['Title'][(df['decade_award']=='1990s')&(df['Revenue (Millions)']>600)]
df['Title'][(df['decade_award']=='2010s')&(df['Revenue (Millions)']>700)]
df['Title'][(df['decade_award']=='2000s')&(df['Revenue (Millions)']>500)]
df['Title'][(df['decade_award']=='1990s')&(df['Revenue (Millions)']<100)&(df['oscar_buzz']>20)]
df['Title'][(df['decade_award']=='1990s')&(df['Revenue (Millions)']>90)&(df['oscar_buzz']==20)]
df['Title'][(df['decade_award']=='2010s')&(df['Revenue (Millions)']>600)&(df['oscar_buzz']<2)]