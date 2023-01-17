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
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('brown')
from nltk.corpus import stopwords
from collections import Counter 
df_train = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', compression='zip',header=0,delimiter='\t',quoting=0, doublequote=False, escapechar='\\')
df_train.shape
df_train.head(5)
df_train['review'][0]
df_train['review'][1]
df_train['review'][2]
def review_to_words(review):
    html_removed = BeautifulSoup(review).get_text()
    punctuation_removed = re.sub('[^a-zA-Z0-9]',' ',str(html_removed))
    lower_case = punctuation_removed.lower()
    words = lower_case.split()
    #Searching sets in python is much faster than searching lists
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops] 
    return( " ".join(words ))
processed_review = review_to_words( df_train['review'][0] )
print(df_train['review'][0])
print(processed_review)
processed_reviews = []
for i in range(0,df_train['review'].size):
    processed_reviews.append(review_to_words(df_train['review'][i]))
    
words = []

for review in processed_reviews:
    for w in review.split():
        words.append(w)

print(words)
print(len(words))
print(len(set(words)))