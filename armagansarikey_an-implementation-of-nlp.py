# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings  

warnings.filterwarnings("ignore")   # ignore warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Sheet_1.csv')

data.head()
data.info()
part1 = data.iloc[:,1:2]

part1
part2 = data.iloc[:,2:3]

part2
comment = pd.concat([part2,part1],axis =1,ignore_index =True) 

comment
comment.columns = ['Response', 'Class']

comment
import re

result = re.sub('[^a-zA-Z]', ' ', comment['Response'][1])

result
result = result.lower()

result
result = result.split()

result
import nltk

from nltk.corpus import stopwords

stopwords_en = stopwords.words('english')

print(stopwords_en)
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
result = [ps.stem(word) for word in result if not word in set(stopwords.words('english'))]

result
result = ' '.join(result)

result
final_result = []

for i in range(80):

    result = re.sub('[^a-zA-Z]', ' ', comment['Response'][i])

    result = result.lower()

    result = result.split()

    result = [ps.stem(word) for word in result if not word in set(stopwords.words('english'))]

    result = ' '.join(result)

    final_result.append(result)
final_result