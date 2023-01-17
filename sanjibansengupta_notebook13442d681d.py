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
import nltk                   #for nlp

from nltk.corpus import  twitter_samples      #for twitter samples

import matplotlib.pyplot as plt 

import random
positive_tweets=twitter_samples.strings('positive_tweets.json')

negative_tweets=twitter_samples.strings('negative_tweets.json')
print(len(positive_tweets))

print(len(negative_tweets))
fig=plt.figure(figsize=(5,5))



labels='Pos','Neg'



sizes=[len(positive_tweets),len(negative_tweets)]



plt.pie(sizes,labels=labels,autopct='%1.1f',shadow=True,startangle=90)



plt.axis('equal')

plt.show()
nltk.download('stopwords')
import re

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer
print()