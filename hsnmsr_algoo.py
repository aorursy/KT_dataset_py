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
inproceedings = pd.read_csv("../input/output_inproceedings.csv",sep=';',low_memory=False).fillna('')

articles = pd.read_csv("../input/output_article.csv",sep=';',low_memory=False).fillna('')

authors1 = inproceedings[["author"]]

authors2 = articles[["author"]]

authors = pd.concat([authors1, authors2])

print (type(authors))

print (authors.head())
from collections import defaultdict

from collections import Counter

import time

import queue
start_time = time.time()

auth_count = defaultdict(int)

for names in authors.author:

    names_parsed = names.split('|')

    for name in names_parsed:

        auth_count[name] += 1

print("--- %s seconds ---" % (time.time() - start_time))
print(type(authors.author))

start_time = time.time()

something = authors.author.str.split('|')

print("--- %s seconds ---" % (time.time() - start_time))

print (something.head())
#for row in articles.values:

#    names = row[1]

#    names_parsed = names.split('|')

#   for name in names_parsed:

#       auth_count[name] += 1