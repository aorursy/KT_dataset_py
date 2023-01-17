import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
netflix_data=pd.read_csv('../input/netflix-shows/netflix_titles.csv')

netflix_data.head()
netflix_data.isnull().sum()
netflix_data.fillna('Unknown', inplace=True)
netflix_data.isnull().sum()
pie_df=netflix_data.groupby('type', axis=0).count()

pie_df['title'].plot(kind='pie',

                     figsize=(7,8),

                     autopct='%1.1f%%',

                    pctdistance=1.12,

                    explode=(0.1,0),

                    colors=['lightcoral', 'darkblue'],

                    labels=None)

plt.legend(labels=pie_df.index, loc='upper left')

plt.title('Distribution of TV shows and Movies')

plt.show()

bar_conti=netflix_data.groupby('country').count()

bar=bar_conti.nlargest(10, 'show_id')

bar['show_id'].plot(kind='bar', figsize=(11,15))

plt.xlabel('Countries')

plt.ylabel('Number of Movies/TV shows')

plt.show()
genere=netflix_data.groupby('listed_in').count()

genere.sort_values(by='show_id', inplace=True, ascending=False)

genere_top=genere.head(20)

genere_top['show_id'].plot(kind='barh', figsize=(11,15))

plt.xlabel('Number of movies/tv shows')

plt.ylabel('Genere')

plt.show


netflix_data.set_index('title', inplace=True)

def get_reccomendation(liked):

        type=netflix_data.loc[liked,'type']

        country=netflix_data.loc[liked,'country']

        genere=netflix_data.loc[liked,'listed_in']



        req=netflix_data[netflix_data['country']==country]

        required=req[req['listed_in']==genere]

        req1=required[required['type']==type]

        return(req1.index.tolist())
liked='Friends'

get_reccomendation(liked)
like='PK'

get_reccomendation(like)