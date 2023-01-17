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
df_train = pd.read_csv("../input/train.csv")
## Starting with EDA!!!
df_train.head(5)
df_train.describe()
df_train.columns
text_features = df_train['comment_text']
text_features.tolist()[0].split()
# Import NLTK libraries to derive usefulness from the comments column

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
# Set Stopwords for English Lnguage

stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(text_features.tolist()[0])
word_tokens
filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence