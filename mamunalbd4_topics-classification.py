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
df = pd.read_csv('/kaggle/input/topics-classification/topics.csv')
df.head(2)
del df['Unnamed: 0']
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(df['question_text'])
dtm
from sklearn.decomposition import NMF

nmf_model = NMF(n_components=9, random_state=101)
nmf_model.fit(dtm)
for index, topic in enumerate(nmf_model.components_):

    print(f"The top 15 model#{index}")

    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])

    print('\n')
topic_model = nmf_model.transform(dtm)
topic_model.argmax(axis =1)
df['mark'] = topic_model.argmax(axis =1)
df.head()
model = {0: 'ecommerce site', 1: 'shipped related', 2: 'product info', 3: 'dress info', 4: 'member promo code', 5: 'problem', 6: 'banking card', 7: 'refund policy', 8: 'cloth info'}
pd.set_option('display.max_colwidth', 200)
df['qus_title'] = df['mark'].map(model)
df.head(20)
plt.figure(figsize=(20,12))

sns.countplot(x = 'qus_title', data =df)