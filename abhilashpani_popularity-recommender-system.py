# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/movies_metadata.csv")
df.head()
df.iloc[0]
m = df['vote_count'].quantile(0.9) # min number of votes to qualify as recommended movie.
C = df['vote_average'].mean()      # average voting in iMdb
print(m) , print(C)
def weighted(x , m=m , C=C):
    v = x['vote_count']
    R = x['vote_average']
    
    return (v/(v+m))*R + (m/(v+m))*C

apply_filter = df.loc[df['vote_count'] >=m]
df.shape , apply_filter.shape
apply_filter['score'] = apply_filter.apply(weighted , axis=1)
apply_filter = apply_filter.sort_values('score' , ascending=False)
apply_filter[['title' , 'vote_count' , 'vote_average' , 'score']].head()
apply_filter[['title' , 'vote_count' , 'vote_average' , 'score']].tail()
