# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/top50spotify2019/top50.csv',sep=',',encoding='latin-1') 
print (data.head(5))
from collections import Counter

pop_artist = collections.Counter(data['Artist.Name'])

pop_artist = pd.DataFrame.from_dict(pop_artist, orient='index').reset_index() 

pop_artist.columns = ['artist', 'freq'] 

pop_artist.sort_values(["freq", "artist"], axis=0, 

                 ascending=False, inplace=True)

print(pop_artist)
pop_genre = collections.Counter(data['Genre'])

pop_genre = pd.DataFrame.from_dict(pop_genre, orient='index').reset_index() 

pop_genre.columns = ['genre', 'freq'] 

pop_genre.sort_values(["freq", "genre"], axis=0, 

                 ascending=False, inplace=True)

print(pop_genre)
import seaborn as sns 

import matplotlib.pyplot as plt 

from scipy.stats import norm 

corrmat = data.corr() 

corrmat[np.abs(corrmat)<.2] = 0

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 



print(corrmat)