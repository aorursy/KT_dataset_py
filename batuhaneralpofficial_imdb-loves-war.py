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
imdb = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv")

#Dataframe
imdb.head()
imdb["Genre"]
genresList = []

for genre in imdb["Genre"]:

    for unique in genre.split(","):

        if unique not in genresList:

            genresList.append(unique)

print(genresList)
genreTotal = {}   #Total Rating

genreCount = {}   #Genre film count



for genre in genresList:

    genreTotal[genre] = 0

    genreCount[genre] = 0

print(genreTotal)

print(genreCount)
for idx, genres in enumerate(imdb["Genre"]):

    #print(idx,genres)

    for genre in genres.split(","):

        genreTotal[genre] += imdb["Rating"][idx]

        genreCount[genre] += 1

print(genreTotal)

print("*****")

print(genreCount)
genreAverage = {}



for genre in genreTotal.keys():

    genreAverage[genre] = genreTotal[genre] / genreCount[genre]

print(genreAverage)
resultGenre = "temp"

resultRating = 0

for genre in genreAverage.keys():

    if genreAverage[genre] > resultRating:

        resultGenre = genre

        resultRating = genreAverage[genre]

        

print("IMDB LOVES " + resultGenre + " WITH AN AVERAGE RATING OF: " + str(round(resultRating,2)))