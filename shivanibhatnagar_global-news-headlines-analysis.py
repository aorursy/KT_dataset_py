# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





df  = pd.read_csv("../input/news-week-aug24.csv")

df.head(5)
df['feed_code'].value_counts()
df[df.feed_code == 'w3-wn']
df.count()
df_no_null = df.dropna()

df_no_null.count()
import nltk

 

englishStopWords = set(nltk.corpus.stopwords.words('english'))

nonEnglishStopWords = set(nltk.corpus.stopwords.words()) - englishStopWords



print ("English Stop words - ",englishStopWords)

print ("Non-English Stop words - ",nonEnglishStopWords)

 
stopWordsDictionary = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}



print(stopWordsDictionary)
df_no_null.headline_text.dropna()
def get_language(text):

    if type(text) is str:

        text = text.lower()

    words = set(nltk.wordpunct_tokenize(text))

    return max(((lang, len(words & stopwords)) for lang, stopwords in stopWordsDictionary.items()), key = lambda x: x[1])[0]
language = get_language(df_no_null['headline_text'][0])

print (df_no_null['headline_text'][0])

print (language)
df_no_null[['headline_text', 'feed_code']][0:1661]
language = []

for row in df_no_null['headline_text']:

    language.append(get_language(row))

language
df_no_null['language'] = language

CountHeadlinesPerLanguage = df_no_null['language'].value_counts()

CountHeadlinesPerLanguage
CountHeadlinesPerLanguage.plot.barh()