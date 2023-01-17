import re

import time

from urllib.request import urlopen



import numpy as np

import pandas as pd
# Load data

train = pd.read_csv('../input/train.csv', usecols=['imdb_id'], dtype=str)

test = pd.read_csv('../input/test.csv', usecols=['imdb_id'], dtype=str)

df = pd.concat([train, test], axis=0).reset_index(drop=True)
# Create new columns for scores and number of ratings

df['avg_rating'] = np.nan

df['num_rating'] = np.nan
def get_imdb_rating(id):

    try:

        html = str(urlopen('http://www.imdb.com/title/'+id+'/ratings').read())

        ix = re.search("vote of ", html).end()

        avg_rating = float(html[ix:ix+100].split()[0])

        ix = re.search("IMDb users have given a", html).start()

        num_rating = float(html[ix-13:ix-2].split()[-1].replace(',', ''))

        return avg_rating, num_rating

    except:

        return np.nan, np.nan
%%time



for i in range(df.shape[0]):

    tid = df['imdb_id'][i]

    avg_rating, num_rating = get_imdb_rating(tid)

    df.loc[i, 'avg_rating'] = avg_rating

    df.loc[i, 'num_rating'] = num_rating

    time.sleep(1.0) #be nice to the servers...
df
# Write scores to file

df.to_csv('imdb_scores.csv', index=False)