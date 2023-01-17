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
df= pd.read_csv('/kaggle/input/tmdb-top-10000-popular-movies-dataset/TMDb_updated.CSV')
df.shape
df.info()
df=df.dropna()
df.shape
df.head()
df.drop(columns='Unnamed: 0',inplace=True)
df.head()
import seaborn as sns
sns.heatmap(df.corr())
df['original_language'].value_counts().head(5).plot(kind='bar')
df1=df.groupby('original_language').sum()
df1[df1['vote_average']>200]
df1
example_overview=df['overview'][0]
example_overview
import re                                  # library for regular expression operations

import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK

from nltk.stem import PorterStemmer  
texts= example_overview.split(' ')
text=[]

for i in texts:

    text.append(i.lower())
stopwords_english = stopwords.words('english') 
words=[]

for word in text: # Go through every word in your tokens list

    if (word not in stopwords_english and  # remove stopwords

        word not in string.punctuation):  # remove punctuation

        words.append(word)



print('removed stop words and punctuation:')

print(words)
# Instantiate stemming class

stemmer = PorterStemmer() 



# Create an empty list to store the stems

stem = [] 



for word in words:

    stem_word = stemmer.stem(word)  # stemming word

    stem.append(stem_word)  # append to the list



print('stemmed words:')

print(stem)