

import numpy as np 

import pandas as pd 



episodes = pd.read_csv('/kaggle/input/st-episodes-imdb/episode.tsv', sep='\t')

ratings = pd.read_csv('/kaggle/input/st-episodes-imdb/ratings.tsv', sep='\t')



series = pd.DataFrame([{'serie':'The Original Series', 'serieId':'TOS', 'tconst':'tt0060028'},

                       {'serie':'The Next Generation', 'serieId':'TNG', 'tconst':'tt0092455'},

                       {'serie':'Deep Space Nine', 'serieId':'DS9', 'tconst':'tt0106145'},

                       {'serie':'Voyager', 'serieId':'VOY', 'tconst':'tt0112178'},

                       {'serie':'Enterprise', 'serieId':'ENT', 'tconst':'tt0244365'}])
episodes['episodeNumber'] = pd.to_numeric(episodes['episodeNumber'], errors = 'coerce', downcast='integer')

epST = episodes[episodes.parentTconst.isin(series.tconst)]

epSTRnk = pd.merge(epST, ratings)

epSTRnk = epSTRnk.sort_values(['parentTconst','seasonNumber','episodeNumber'])

epSTRnk['episode'] = epSTRnk.groupby(['parentTconst']).cumcount()

epSTRnk

Result = epSTRnk.merge(series, left_on='parentTconst', right_on='tconst')[['serieId','serie','seasonNumber','episodeNumber','episode','averageRating','numVotes']]

Result.to_csv('epSTRnk.csv', index='false')