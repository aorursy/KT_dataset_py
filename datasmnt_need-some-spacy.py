# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
nlp = spacy.load("en_core_web_sm")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
train_df.describe()
train_df.info()
train_df.isnull().sum()
train_df.head()
train_df = train_df.fillna('')
def pos_exraction(text,on = 'noun'):
    if on == 'noun':
        doc = nlp(text)
        nouns = [chunk.text for chunk in doc.noun_chunks]
        return nouns
    elif on == 'verbs':
        doc = nlp(text)
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        return verbs
train_df['texts_verb'] = train_df['text'].apply(lambda x: pos_exraction((x),on = 'verbs'))
train_df['texts_nouns'] = train_df['text'].apply(lambda x: pos_exraction((x),on = 'noun'))
train_df
