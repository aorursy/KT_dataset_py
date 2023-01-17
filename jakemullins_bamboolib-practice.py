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
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
!pip install --upgrade bamboolib>=1.4.1
import bamboolib as bam

import pandas as pd



movies = pd.read_csv("../input/IMDB-Movie-Data.csv")

movies = movies.loc[movies['Metascore'].notna()]

movies['Genre'] = movies.Genre.str.split(',').str[0]

movies['Normalized Polarization Factor'] =  movies.Metascore/100 - movies.Rating/10



crowd_pleasers = movies[movies['Normalized Polarization Factor'] < 0]

percent_crowd_pleasers = '{}%'.format( (crowd_pleasers.shape[0] / 1000) * 100 )





highbrow_films = movies[movies['Normalized Polarization Factor'] > 0] 

percent_highbrow_films = '{}%'.format( (highbrow_films.shape[0] / 1000) * 100 )





crowd_pleasers = crowd_pleasers.groupby(['Genre']).agg({'Normalized Polarization Factor': ['mean', 'min']})

crowd_pleasers.columns = ['_'.join(multi_index) for multi_index in crowd_pleasers.columns.ravel()]

crowd_pleasers = crowd_pleasers.sort_values(by=['Normalized Polarization Factor_mean'], ascending=[False])

crowd_pleasers = crowd_pleasers.reset_index()



highbrow_films = highbrow_films.groupby(['Genre']).agg({'Normalized Polarization Factor': ['mean', 'max']})

highbrow_films.columns = ['_'.join(multi_index) for multi_index in highbrow_films.columns.ravel()]

highbrow_films = highbrow_films.sort_values(by=['Normalized Polarization Factor_mean'], ascending=[True])

highbrow_films = highbrow_films.reset_index()



movies

crowd_pleasers

highbrow_films
