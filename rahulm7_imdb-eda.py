# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

os.chdir("../input")

os.listdir()
imdb = pd.read_csv("movie_metadata.csv")
imdb.columns
imdb.info()
imdb.shape
# No. of unique director

len(imdb['director_name'].unique())
# Top 5 grosses by a director

imdb[['director_name','gross']].groupby('director_name').sum().sort_values(by='gross',ascending=False).head(5)
# Highest grossing movie by each director

# Split-apply-combine

def ranker(df):

    df['movie_rank'] = np.arange(len(df))+1

    return df



imdb.sort_values('gross',ascending=False,inplace=True)

imdb = imdb.groupby('director_name').apply(ranker)

imdb[imdb['movie_rank']==1].head(10)