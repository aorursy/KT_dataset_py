# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

imdbData=pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv", encoding='latin-1', error_bad_lines=False)

englishWords=pd.read_csv("../input/english/EnglishWords.csv", encoding='latin-1', error_bad_lines=False)

np.random.seed(1)

# Any results you write to the current directory are saved as output.
x=[]

review=[]

imdbRating=[]

englishWordsList=[]
for i in imdbData['review'].fillna(''):

    review.append(i)
for i in englishWords['English:'].fillna(''):

    englishWordsList.append(i)
appendList=[]

randomIndexesForReviews=np.random.randint(0, len(review)-1, 1009)

randomIndexesForEnglishWords=np.random.randint(0, len(englishWords)-1, 10000)

shortenedReviews=[]

shortenedWords=[]

for i in randomIndexesForReviews:

    shortenedReviews.append(review[i])

for i in randomIndexesForEnglishWords:

    shortenedWords.append(englishWordsList[i])

for i in shortenedReviews:

    appendList=[0]*len(shortenedWords)

    splitWords=i.split()

    for iterator in splitWords:

        try:

            appendList[shortenedWords.index(iterator)]=splitWords.count(iterator)

        except:

            pass

    x.append(appendList)
rankings=imdbData['sentiment'].fillna('')

for i in randomIndexesForReviews:

    sign=0

    if rankings[i]=='negative':

        sign=1

    imdbRating.append(sign)
y=[]

for i in imdbRating:

    y.append(i)
print(x)
print(y)