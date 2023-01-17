import os

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

import json

from tqdm import tqdm_notebook

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import mean_absolute_error

from scipy import sparse

import pyLDAvis.gensim

import gensim

from gensim.matutils  import Sparse2Corpus

from gensim.corpora import Dictionary

from gensim.models import LdaModel

from sklearn.linear_model import Ridge
from html.parser import HTMLParser



class MLStripper(HTMLParser):

    def __init__(self):

        self.reset()

        self.strict = False

        self.convert_charrefs= True

        self.fed = []

    def handle_data(self, d):

        self.fed.append(d)

    def get_data(self):

        return ''.join(self.fed)



def strip_tags(html):

    s = MLStripper()

    s.feed(html)

    return s.get_data()
PATH_TO_DATA = '../input/'
def read_json_line(line=None):

    result = None

    try:        

        result = json.loads(line)

    except Exception as e:      

        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      

        new_line = list(line)

        new_line[idx_to_replace] = ' '

        new_line = ' '.join(new_line)     

        return read_json_line(line=new_line)

    return result



def preprocess(path_to_inp_json_file):

    output_list = []

    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:

        for line in tqdm_notebook(inp_file):

            json_data = read_json_line(line)

            content = json_data['content'].replace('\n', ' ').replace('\r', ' ')

            content_no_html_tags = strip_tags(content)

            output_list.append(content_no_html_tags)

    return output_list
train_raw_content = preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 'train.json'),)
test_raw_content = preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA,  'test.json'),)
cv = CountVectorizer(max_features=10000, min_df = 0.1, max_df = 0.8)

sparse_train = cv.fit_transform(train_raw_content)

sparse_test  = cv.transform(test_raw_content)
full_sparse_data =  sparse.vstack([sparse_train, sparse_test])
train_target = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'), 

                           index_col='id')
y_train = train_target['log_recommends'].values
#Transform our sparse_data to corpus for gensim

corpus_data_gensim = gensim.matutils.Sparse2Corpus(full_sparse_data, documents_columns=False)
#Create dictionary for LDA model

vocabulary_gensim = {}

for key, val in cv.vocabulary_.items():

    vocabulary_gensim[val] = key

    

dict = Dictionary()

dict.merge_with(vocabulary_gensim)
lda = LdaModel(corpus_data_gensim, num_topics = 30 )
data_ =  pyLDAvis.gensim.prepare(lda, corpus_data_gensim, dict)

# pyLDAvis.display(data_)
def document_to_lda_features(lda_model, document):

    topic_importances = lda.get_document_topics(document, minimum_probability=0)

    topic_importances = np.array(topic_importances)

    return topic_importances[:,1]



lda_features = list(map(lambda doc:document_to_lda_features(lda, doc),corpus_data_gensim))
data_pd_lda_features = pd.DataFrame(lda_features)

data_pd_lda_features.head()
data_pd_lda_features_train = data_pd_lda_features.iloc[:y_train.shape[0]]

data_pd_lda_features_train['target'] = y_train



fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(20.7, 8.27)

sns.heatmap(data_pd_lda_features_train.corr(method = 'spearman'), cmap="RdYlGn", ax = ax)
X_tr = sparse.hstack([sparse_train, data_pd_lda_features_train.drop('target', axis = 1)]).tocsr()
X_test = sparse.hstack([sparse_test, data_pd_lda_features.iloc[y_train.shape[0]:]]).tocsr()
ridge = Ridge(random_state=17)

ridge.fit(X_tr,y_train)
subm = ridge.predict(X_test)
plt.hist(subm, bins=30, alpha=.5, color='green', label='pred', range=(0,10));

plt.legend();