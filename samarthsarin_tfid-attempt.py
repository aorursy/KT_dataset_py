# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import spacy

from nltk.stem import WordNetLemmatizer

nlp = spacy.load('en_core_web_sm')



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
df.head()
df.drop('qid',1,inplace = True)
lem = WordNetLemmatizer()
def cleaning(text):

    words = []

    doc = nlp(text)

    for token in doc:

        if not token.is_punct:

            words.append(lem.lemmatize(token.text))

    return ' '.join(words)

    
df['question_text'] = df['question_text'].apply(cleaning)