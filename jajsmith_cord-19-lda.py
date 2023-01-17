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

import numpy as np

import os

import json

import glob

import sys

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#sys.path.insert(0, "../")
!ls ../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/
root_path = '../input/CORD-19-research-challenge'

json_filenames = glob.glob(root_path+'/**/*.json', recursive=True)

print('Number of files: ', len(json_filenames))
def load_df(files, df):

    for fname in files:

        

        row = {"doc_id": None, "source": None, "title": None,

              "abstract": None, "text_body": None}

        

        with open(fname) as json_data:

            data = json.load( json_data )

            

            row['doc_id'] = data['paper_id']

            row['title'] = data['metadata']['title']

            

            abstract_list = [a['text'] for a in data['abstract']]

            abstract = "\n ".join(abstract_list)

            row['abstract'] = abstract

            

            body_list = [b['text'] for b in data['body_text']]

            body = "\n ".join(body_list)

            

            row['text_body'] = body

            

            row['source'] = fname.split('/')[3]

            

            df = df.append(row, ignore_index=True)

            

    return df
features = {"doc_id": [None], "source": [None], "title": [None],

                  "abstract": [None], "text_body": [None]}

df = pd.DataFrame.from_dict(features)



df = load_df( json_filenames, df )
df.shape
df.head(5)
csv = df.to_csv('kaggle_cord19_text.csv')
# If reloading

df = pd.read_csv('/kaggle/working/kaggle_cord19_text.csv')

df = df.dropna()
n_words = 1000



tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_words, stop_words='english')

tf = tf_vectorizer.fit_transform( df['title'] )

tf_feature_names = tf_vectorizer.get_feature_names()
tf_feature_names[550:555]
n_topics = 10



lda = LatentDirichletAllocation(n_components=n_topics, 

    max_iter=5, 

    learning_method='online',

    learning_offset=50.,

    random_state=0).fit(tf)
n_top_words = 20

for topic_i, topic in enumerate(lda.components_):

    print( "Topic ", topic_i )

    print( "Words: ", [tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] )
def print_lda_topics(data, n_words, n_topics, n_top_words):

    print('Generating word features...')

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_words, stop_words='english')

    tf = tf_vectorizer.fit_transform( data )

    tf_feature_names = tf_vectorizer.get_feature_names()

    

    print('Fitting LDA model...')

    lda = LatentDirichletAllocation(n_components=n_topics, 

        max_iter=5, 

        learning_method='online',

        learning_offset=50.,

        random_state=0).fit(tf)

    

    for topic_i, topic in enumerate(lda.components_):

        print( "Topic ", topic_i )

        print( "Words: ", [tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] )

    

    return tf, tf_vectorizer, lda
print_lda_topics( df['abstract'], 1000, 10, 10)
print_lda_topics( df['text_body'], 10000, 20, 20)