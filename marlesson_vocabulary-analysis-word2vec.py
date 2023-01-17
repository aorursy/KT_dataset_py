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
#Load required libraries
import numpy as np
import pandas as pd
#For displaying complete rows info
pd.options.display.max_colwidth=500
import tensorflow as tf
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
print(tf.__version__)

#Load data into pandas dataframe
df=pd.read_csv("../input/articles.csv",encoding="utf8")
df.head(2)
print(df["title"][0],"\n",df["text"][0])
#Properly formatted data removing nans
df.drop_duplicates(subset=["text"],inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
import gensim
import string
import re
articles_tokens=[]
for i in range(len(df["text"])):
    articles_tokens.append([word for word in word_tokenize(str(df["text"][i].lower())) if len(word)>2])
model = gensim.models.Word2Vec(articles_tokens, min_count=5,size=100,workers=4)
model.wv.most_similar("lula")

model.wv.most_similar("propina")
model.wv.most_similar("esporte")
