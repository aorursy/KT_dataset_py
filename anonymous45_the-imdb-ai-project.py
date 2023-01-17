#Anisha



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

EnglishWords = pd.read_csv("../input/english/EnglishWords.csv")

IMDBDataset = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

random.seed(1)

# Any results you write to the current directory are saved as output.
#Grace

#create arrays 

x=[]

reviews=[]

imdbRating=[]

englishWordList=[]

#Anushka

for var in IMDBDataset['review'].fillna(''):   #Will explain in class about review

        reviews.append(var)
#Ashna

for i in EnglishWords['English:'].fillna(''):                       

    englishWordList.append(i)

#Devang

appendList=[]

randomIndexesForReviews=np.random.randint(0, len(reviews)-1, 1009)

randomIndexesForEnglish=np.random.randint(0,len(englishWordList)-1, 10000)

shortenedReviews=[]

shortenedWords=[]

for i in randomIndexesForReviews:

    shortenedReviews.append(reviews[i])

for i in randomIndexesForEnglish:

    shortenedWords.append(englishWordList[i])

for i in shortenedReviews:

    appendList=len(shortenedWords)*([0])

    splitWords=i.split()

    for iterator in splitWords:

        try:

            appendList[shortenedWords.index(iterator)]=splitWords.count(iterator)

        except:

            pass

    x.append(appendList)

    
#Devang

rankings=IMDBDataset['sentiment'].fillna('')

for i in randomIndexesForReviews:

    sign=0

    if rankings[i]=='negative':

        sign=1

    imdbRating.append(sign)
#Akhil

y=[]

for i in imdbRating:  #Explain this whole block and coding conventions

    y.append(i)











#Krish

print(x) 



#x is whatever the variable is set to be

#Sareen

print(y)
