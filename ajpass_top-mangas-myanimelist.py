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
df = pd.read_csv('/kaggle/input/top-ranked-mangas-myanimelist-mal/top500mangaMAL.csv')

df.head()
df.info()
# Get int columns, show head

df.select_dtypes(include='int').head()
# Get object columns, show head

df.select_dtypes(include='object').head()
# Get float columns, show head

df.select_dtypes(include='float').head()
hasEnglishTitle = df[df['English Title'] != 'Unknown'] 

hasEnglishTitle.Members.mean()

hasNotEnglishTitle = df[df['English Title'] == 'Unknown'] 

hasNotEnglishTitle.Members.mean()

firstRowGenres = df['Genres'][0]

firstRowGenres
import ast

firstRow = ast.literal_eval(firstRowGenres)

firstRow[0]
secondRowDates = df['Published Dates'][1]

secondRowDates
datesDict  = ast.literal_eval(secondRowDates)

datesDict['from']
datesDict['to']
all_genres = []



for row in df['Genres']:

    all_genres.append(ast.literal_eval(row))
len(all_genres)
genresCountDict = {}

all_genres[10]
for row in all_genres:

    for genre in row:

        if genre in genresCountDict:

            genresCountDict[genre] += 1

        else:

            genresCountDict[genre] = 1

genresCountDict
genresDF = pd.DataFrame.from_dict(genresCountDict, orient='index') # Dictionary to DataFrame

genresDFsorted = genresDF.sort_values(by=0, ascending=False) # sort by descending



genresDFsorted
import seaborn as sns

import matplotlib.pyplot as plt

print('Setup complete!')

plt.figure(figsize=(15,10))

sns.barplot(x=genresDFsorted.index, y=genresDFsorted[0])

plt.title("Popular Genres in top 500 ranked mangas")

plt.xlabel('Genres')

plt.ylabel('Manga Count')

plt.xticks(rotation=65)

plt.show()
import plotly.express as px



fig = px.bar(x=genresDFsorted.index, y=genresDFsorted[0], text=genresDFsorted[0], labels={'x':'Genres', 'y':'Manga Count'})

fig.show()
import pandas_profiling as pp

pp.ProfileReport(df)