# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



import pandas as pd

# pandas defaults

pd.options.display.max_columns = 500

pd.options.display.max_rows = 500



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
imdb = pd.read_csv("../input/IMDB-Movie-Data.csv")
# top 5 rows

imdb.head()
#renaming some cols

imdb.rename(columns = {'Revenue (Millions)':'Rev_M','Runtime (Minutes)':'Runtime_min'},inplace=True)
imdb['AvgRating'] = (imdb['Rating'] + imdb['Metascore']/10)/2
def custom_rating(genre,rating):

    if 'Thriller' in genre:

        return min(10,rating+1)

    elif 'Comedy' in genre:

        return max(0,rating-1)

    else:

        return rating

        

imdb['CustomRating'] = imdb.apply(lambda x: custom_rating(x['Genre'],x['Rating']),axis=1)
# Single condition: dataframe with all movies rated greater than 8

imdb_gt_8 = imdb[imdb['Rating']>8]



imdb_gt_8.head()
# Multiple conditions: AND - dataframe with all movies rated greater than 8 and having more than 100000 votes



And_imdb = imdb[(imdb['Rating']>8) & (imdb['Votes']>100000)]



And_imdb.head()
# Multiple conditions: OR - dataframe with all movies rated greater than 8 or having a metascore more than 90



Or_imdb = imdb[(imdb['Rating']>8) | (imdb['Metascore']>80)]

Or_imdb.head()

# Multiple conditions: NOT - dataframe with all emovies rated greater than 8 or having a metascore more than 90 have to be excluded



Not_imdb = imdb[~((imdb['Rating']>8) | (imdb['Metascore']>80))]

Not_imdb.head()
# Single condition: dataframe with all movies rated greater than 8

imdb_gt_8 = imdb[imdb['Rating']>8]



# Multiple conditions: AND - dataframe with all movies rated greater than 8 and having more than 100000 votes

And_imdb = imdb[(imdb['Rating']>8) & (imdb['Votes']>100000)]



# Multiple conditions: OR - dataframe with all movies rated greater than 8 or having a metascore more than 90

Or_imdb = imdb[(imdb['Rating']>8) | (imdb['Metascore']>80)]



# Multiple conditions: NOT - dataframe with all emovies rated greater than 8 or having a metascore more than 90 have to be excluded

Not_imdb = imdb[~((imdb['Rating']>8) | (imdb['Metascore']>80))]
new_imdb = imdb[len(imdb['Title'].split(" "))>=4]

#create a new column

imdb['num_words_title'] = imdb.apply(lambda x : len(x['Title'].split(" ")),axis=1)

#simple filter on new column

new_imdb = imdb[imdb['num_words_title']>=4]

new_imdb.head()
new_imdb = imdb[imdb.apply(lambda x : len(x['Title'].split(" "))>=4,axis=1)]

new_imdb.head()
year_revenue_dict = imdb.groupby(['Year']).agg({'Rev_M':np.mean}).to_dict()['Rev_M']

def bool_provider(revenue, year):

    return revenue<year_revenue_dict[year]

    

new_imdb = imdb[imdb.apply(lambda x : bool_provider(x['Rev_M'],x['Year']),axis=1)]



new_imdb.head()
from tqdm import tqdm, tqdm_notebook

tqdm_notebook().pandas()



new_imdb['rating_custom'] = imdb.progress_apply(lambda x: custom_rating(x['Genre'],x['Rating']),axis=1)

new_imdb.head()