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
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))
print("", stopWords)
wordList= open("../input/wordlistrelevante/WordList-relevante.txt", "r")

print(wordList.read())
file1 = open("../input/wordlistrelevante/WordList-relevante.txt")

line = file1.read() # Use this to read file content as a stream: 

words = line.split() 

print(words)

filtered_sentence = []

#for r in words: 

#    if not r in stopWords: 

#      filtered_sentence.append(r)



for r in words: 

    if not r in stopWords: 

        appendFile = open('filteredtext.txt','a') 

        appendFile.write(" "+r) 

        appendFile.close() 

    

#print(filtered_sentence)
filteredText = open("filteredtext.txt")

print(filteredText.read())