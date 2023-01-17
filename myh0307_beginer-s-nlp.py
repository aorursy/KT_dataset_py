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
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df.head()
import re

from bs4 import BeautifulSoup



def trim_text(data):

    trim_words = BeautifulSoup(data).get_text()

    trim_words = re.sub('[^a-zA-Z]',' ', trim_words)

    return trim_words



train_df['trim_text']=train_df['text'].apply(trim_text)

test_df['trim_text']= test_df['text'].apply(trim_text)
train_df.head()
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



words_lem = WordNetLemmatizer()

stemmer = PorterStemmer()



def word_review(data):

    words = data.lower().split()

    stops = set(stopwords.words('english')) # increase in processing speeds

    words = [w for w in words if not w in stops] # remove stop words

    lem_words = [words_lem.lemmatize(w) for w in words]

    stem_words =[stemmer.stem(w) for w in lem_words]

    return (' '.join(stem_words)) # return clean sentences



train_df['clean_join_text']=train_df['trim_text'].apply(word_review)

test_df['clean_join_text'] = test_df['trim_text'].apply(word_review)
train_df.head()
train_df.drop(['trim_text'], axis=1, inplace=True)

test_df.drop(['trim_text'], axis=1, inplace=True)
train_df.shape, test_df.shape
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

%matplotlib inline



def display_wc(data):

    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=600).generate(data)

    plt.figure(figsize=(10,5))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



display_wc(' '.join(train_df[train_df['target']==1]['clean_join_text']))

display_wc(' '.join(train_df[train_df['target']==0]['clean_join_text']))



display_wc(' '.join(test_df['clean_join_text']))


# number of words check

train_df['n_words']=train_df['clean_join_text'].apply(lambda x: len(str(x).split()))

train_df['n_unq_words']=train_df['clean_join_text'].apply(lambda x: len(set(str(x).split())))



test_df['n_words']=test_df['clean_join_text'].apply(lambda x: len(str(x).split()))

test_df['n_unq_words']=test_df['clean_join_text'].apply(lambda x: len(set(str(x).split())))



# target :1, vs. target :0



import seaborn as sns



fig, axes = plt.subplots(2,2)

fig.set_size_inches(10,12)

sns.distplot(train_df[train_df['target']==1]['n_words'],bins=50, ax=axes[0][0])

axes[0][0].axvline(train_df[train_df['target']==1]['n_words'].median(), linestyle='dashed')

axes[0][0].set_title('dist target:1 n_words')

sns.distplot(train_df[train_df['target']==0]['n_words'],bins=50, ax=axes[0][1], color='pink')

axes[0][1].axvline(train_df[train_df['target']==0]['n_words'].median(), linestyle='dashed')

axes[0][1].set_title('dist target:0 n_words')

sns.distplot(train_df[train_df['target']==1]['n_unq_words'],bins=50, ax=axes[1][0])

axes[1][0].axvline(train_df[train_df['target']==1]['n_unq_words'].median(), linestyle='dashed')

axes[1][0].set_title('dist target:1 n_unq_words')

sns.distplot(train_df[train_df['target']==0]['n_unq_words'],bins=50, ax=axes[1][1],color='pink')

axes[1][1].axvline(train_df[train_df['target']==0]['n_unq_words'].median(), linestyle='dashed')

axes[1][1].set_title('dist target:1 n_unq_words')



def simp_eda(data):

    print(data[data['target']==1]['n_words'].describe())

    print(data[data['target']==0]['n_words'].describe())

    print(data[data['target']==1]['n_unq_words'].describe())

    print(data[data['target']==0]['n_unq_words'].describe())



simp_eda(train_df)
from collections import Counter

dataset=(' '.join(train_df['clean_join_text']))

sep = dataset.split()

len(Counter(sep))
from sklearn.feature_extraction.text import CountVectorizer

vect=CountVectorizer(analyzer='word', max_features=5000) # use 5000 for max feature

train_data_features=vect.fit_transform(train_df['clean_join_text'])

test_data_features = vect.transform(test_df['clean_join_text'])



word_name= vect.get_feature_names() #check the classified words

#print(word_name)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



train_x, val_x, train_y, val_y = train_test_split(train_data_features, train_df['target'], train_size=0.9, test_size=0.1, random_state=0)



rf = RandomForestClassifier(n_estimators=100)

rf.fit(train_x, train_y)

rf_result=rf.predict(val_x)

accuracy_score(val_y, rf_result)
from sklearn.linear_model import LogisticRegression



lg = LogisticRegression()

lg.fit(train_x,train_y)

lg_result=lg.predict(val_x)



accuracy_score(val_y, lg_result)