# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/nlp-topic-modelling/Reviews.csv')
df.head()
df = df[['Text']]
pd.set_option('display.max_colwidth', 200)

df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(df['Text'])
dtm
from sklearn.decomposition import NMF
nmf_model = NMF(n_components=9, random_state=101)
nmf_model.fit(dtm)
for index, topic in enumerate(nmf_model.components_):

    print(f"The new 15 models# {index}")

    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])

    print('\n')
topic_model = nmf_model.transform(dtm)
topic_model.argmax(axis =1)
df['topic'] = topic_model.argmax(axis =1)
df.head(10)
model = {0:'sweet related', 1:'coffee related', 2:'product review', 3:'tea related', 4:'animal food', 5:'shopping Related', 6:'food love', 7:'food related', 8:'cookie & chocolate'}
df['Title'] = df['topic'].map(model)
df.head()
df['Title'].value_counts()
plt.figure(figsize=(15,9))

sns.countplot(x = 'Title', data = df)