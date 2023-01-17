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
import bs4

from bs4 import BeautifulSoup as soup

from urllib.request import urlopen

import string

from nltk.corpus import stopwords, words

from nltk.corpus import wordnet as wn

import pandas as pd
news_url = "https://news.google.com/rss"

Client = urlopen(news_url)

xml_page = Client.read()

Client.close()



soup_page = soup(xml_page, "xml")

news_list = soup_page.findAll("item")



newslist = []



for news in news_list:

    cleantext = soup(news.description.text, "lxml").text

    newslist.append(cleantext)
newslist[1]
def text_process(title):

    

    nop = [char for char in title if char not in string.punctuation]

    

    nop = ''.join(nop)

    

    return [word for word in nop.split() if word in word.lower() not in stopwords.words('english')]





def word_list(wordlist):

    """"  

     Return only word form the news sentences

    """

    li = []

    for word in wordlist:

        if word in words.words():

            li.append(word)

    return li
finalword = []

for w in newslist:

    word = text_process(w.lower())

    finalword.append(word_list(word))
finalword
# Make 2D array to 1D Array



from itertools import chain

flatten_list = list(chain.from_iterable(finalword))

length = len(flatten_list)

length
repeatwordDict = {}

repeatwordList = []

count = 1



for w in flatten_list:

    if w not in repeatwordList:

        repeatwordDict[w] = count

    else:

        repeatwordDict[w] = repeatwordDict[w] + 1

    

print(len(repeatwordDict))



orderDict = sorted(repeatwordDict.items(), key = lambda x: x[1], reverse=True)
topwords = []

wordDict = {}



for w in orderDict: 

    w = w[0]

    try:

        syns = wn.synsets(w)

        tmp = wn.synsets(w)[0].pos()

        if tmp == 'a' or tmp == 's': 

            wordDict = {

                'Name': w,

                'Type': 'Adjective',

                'Defination': syns[0].definition(),

                'Example': syns[0].examples()

            }

            topwords.append(wordDict)

        if tmp == 'r':

            wordDict = {

                'Name': w,

                'Type': 'Adverb',

                'Defination': syns[0].definition(),

                'Example': syns[0].examples()

            }

            topwords.append(wordDict)

        if tmp == 'v':

            wordDict = {

                'Name': w,

                'Type': 'Verb',

                'Defination': syns[0].definition(),

                'Example': syns[0].examples()

            }

            topwords.append(wordDict)

    except:

        print('Error')
Engwords = pd.DataFrame(topwords)

Engwords = Engwords.sort_values('Name', ignore_index=True)

Engwords.head()
Engwords.to_csv('words.csv', index=False)