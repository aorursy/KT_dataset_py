# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import visualising libraries
import matplotlib.pyplot as plt
import seaborn as sn
bollywood = pd.read_csv('../input/bollywood-movies-analysis/bollywood.csv', encoding= 'unicode_escape')
actors = pd.read_csv('../input/bollywood-movies-analysis/bollywood-actors.csv')
actress = pd.read_csv('../input/bollywood-movies-analysis/bollywood-actress.csv', encoding= 'unicode_escape')
bollywood.head(5)
bollywood.tail(5)
bollywood.shape
bollywood.Year.value_counts()
no_of_movies_per_year = bollywood.Year.value_counts().reset_index()
no_of_movies_per_year.columns = ['year','No of Movies']
# pd dataframe between years and number of movies released
no_of_movies_per_year.head(10)
x = no_of_movies_per_year.iloc[0:1]
x
x.append(no_of_movies_per_year.iloc[-1]) 
# The first entry tells the year with highest number of movies released
# The second entry tells the year with lowest number of movies released
no_of_movies_per_year.shape
no_of_movies_per_year.sort_values('year')
sn.barplot(x='year', y = 'No of Movies', data = no_of_movies_per_year)
sn.boxplot(no_of_movies_per_year['No of Movies'])
bollywood.head(5)
directors = bollywood.Director.value_counts().reset_index()
directors
directors.iloc[0:4]
b = bollywood
b.head(5)
b = b[b['Year']>1998][['Year','Director']]
b
b.shape
d = b.Director.value_counts().reset_index()
d
d.iloc[0:4]
# Top 3 directors in the last decade based on number of movies released
actors.head(5)
a = actors.sort_values('Height(in cm)', ascending=False)
a
x = a.iloc[0:1]
x = a[a['Height(in cm)']==188][['Name','Height(in cm)']]
x
x.append(a[a['Height(in cm)']==163][['Name','Height(in cm)']])

# List of tallest and shortest actors
actress.head(5)
a = actress.sort_values('Height(in cm)', ascending=False)
a
x = a.iloc[0:1]
x
x.append(a[a['Height(in cm)']==152][['Name','Height(in cm)']])

# List of tallest and shortest actors
