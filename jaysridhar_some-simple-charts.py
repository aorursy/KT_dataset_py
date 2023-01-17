import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt



x = pd.read_csv('../input/movie_metadata.csv')



def annHbars(a):

    for idx, val in enumerate(a):

        if val < 100:

            color = 'black'

            halign = 'left'

            hpad = 5

        else:

            color = 'blue'

            halign = 'right'

            hpad = -5

        ax.text(val + hpad, idx, "{}".format(val), color=color, verticalalignment='center', horizontalalignment=halign, weight='heavy', fontsize=10)



x['title_year'] = x.title_year.fillna(0).astype(int)

for a in ['num_critic_for_reviews', 'director_facebook_likes', 'actor_1_facebook_likes', 'gross']:

    x[a] = x[a].fillna(0).astype(int)

for a in ['movie_title', 'director_name']:

    x[a] = x[a].str.replace('\xc2\xa0', ' ')

x = x.drop(x[x.title_year == 0].index)

x['gross_mn'] = (x['gross']/1000000).astype(int)

x['mname_yr'] = x['movie_title'].str.cat(x['title_year'].apply(str), sep = ',')

x['dname_mname'] = x['director_name'].str.cat(x['movie_title'], sep=' (').str.cat(x['title_year'].apply(str), sep='| ') + ')'

data = x.sort_values('gross',ascending=False).head(25)[['movie_title', 'director_name', 'dname_mname', 'gross_mn']].drop_duplicates().sort_values('gross_mn')

data = data.set_index('dname_mname')['gross_mn']

ax = data.plot(kind='barh', color='pink', grid=True, title='Movie Gross in Millions', width=0.8, figsize=(7,12))

ax.set_xlabel('Gross in Millions ($)')

ax.set_ylabel('')

annHbars(data)

print (x.head())