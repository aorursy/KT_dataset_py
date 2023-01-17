# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pandas as pd

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
#Number of articles

len(df)
sars_cov = []

all_words = []

for i in range(len(df)):

    all_text = str(str(df.iloc[i].title) + ' '+str(df.iloc[i].abstract)).split()

    all_words.append(all_text)

    #print (i,all_text)

    if 'SARS' in all_text or 'CoV' in all_text or 'COV' in all_text:

        sars_cov.append(i)

print ('Total number, few numeric indices', len(sars_cov), sars_cov[0:5] )    

find_words = ['success','time']
priority = []

for ii,i in enumerate(all_words):

    if True in np.in1d(i, find_words):

        priority.append(ii)
# First 5 priority papers

for i in  priority[0:5]:

    print(i, df.iloc[i].title)

    print ()
from collections import Counter

concat = np.concatenate(np.array(all_words).ravel())

#Let's exclude some common words

common_words = ['a', 'A', 'an', 'An', 'the', 'The', 'is','than', 'against','other', 'are', 'to', 'and', 'of', 'by', 'as', 'We', 'using', 'may', 'not', 'these', 'been', 'This', 'During', 'during', 'their','but', 'into','its','both','that', 'from', 'which', 'have', 'has', 'be', 'in', 'for', 'with', 'were', 'was', 'or', 'this', 'In', 'at', '','there' 'as', 'on', 'we', 'can', 'between', 'also']

filtered = list(filter(lambda a: a not in common_words, concat))

Counter = Counter(filtered)

most_occur = Counter.most_common(100) 

  

print(most_occur) 
import matplotlib.pyplot as plt

%matplotlib inline

labels, values = zip(*most_occur[0:20])

indexes = np.arange(len(labels))

width = .2

plt.barh(indexes, values, width)

plt.yticks(indexes + width * 0.5, labels)

plt.gca().invert_yaxis()

plt.show()