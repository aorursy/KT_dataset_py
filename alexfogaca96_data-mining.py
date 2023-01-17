# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir('../input'))

# Any results you write to the current directory are saved as output.

TEXT_DATA_DIR = '../input/'
orderedList = sorted(os.listdir(TEXT_DATA_DIR))
texts = []
for name in orderedList:
    path = TEXT_DATA_DIR + name
    with open(path, 'r', encoding='latin-1') as content_file:
        texts.append(content_file.readlines())
    
print('Found %s texts.' % len(texts))


# Feature extraction from text
# Method: bag of words 
# https://pythonprogramminglanguage.com
 
vectorizer = CountVectorizer()
for i in range(0, len(texts)):
    print ()
    print ( orderedList[i] )
    print ( 'vectors: ', vectorizer.fit_transform(texts[i]).todense().shape )
    vocabulary = vectorizer.vocabulary_
    print ( dict(list (vocabulary.items())[0:10]) )
    




























































































