# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from bs4 import BeautifulSoup

import requests

from tqdm import tqdm

from collections import Counter

from nltk.corpus import stopwords 

from tqdm import tqdm
string = 'abcdefghijklmnopqrstuvwxyz'

length = len(string)

dictionary = []

for i in tqdm(range(length)):    

    url = 'https://musicterms.artopium.com/'+string[i]+'/index.htm'

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER'}

    res = requests.get(url,headers=headers)

    soup = BeautifulSoup(res.content,'lxml')

    for line in soup.find_all(class_='listcol'):

        dictionary += line.get_text().split('\n')
dictionary_ = []

for word in dictionary:

    if len(word)>0:

        dictionary_.append(word.lower())
len(dictionary_)
data = pd.DataFrame({'word':dictionary_})

data.head()
data.to_csv('data.csv')
# text = []

# for title in soup.find_all('p',class_='paper-item'):

#     [s.extract() for s in title('em')]

#     content = title.get_text().split(' ')

#     content = [word for word in content if not word in stop_words ]

#     text.append(content)
# len(word_counter)
# word_counter = Counter([word for title in text for word in title])

# print(len(word_counter))

# word_counter.most_common(50)
# url = 'https://acl2018.org/programme/papers/'

# headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER'}

# res = requests.get(url,headers=headers)

# soup = BeautifulSoup(res.content,'lxml')
# text_2018 = []

# for title in soup.find_all('span',class_='paper-title'):

#     content = title.get_text().split(' ')

#     text_2018.append([word for word in content if not word in stop_words ])
# word_counter_ = Counter([word for title in text_2018 for word in title])

# print(len(word_counter))

# word_counter_.most_common(50)
# word_dict = dict()

# for word,num in word_counter.most_common(150):

#     word_dict[word] = word_counter[word] - word_counter_[word]
# sorted(word_dict,reverse=True)