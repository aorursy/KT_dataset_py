! conda install -y -c conda-forge hvplot=0.5.2 bokeh=1.4.0 gensim umap-learn
from pathlib import Path

import os

import json

from string import punctuation

import re

import warnings

import string



import numpy as np

import pandas as pd

import dask.dataframe as dd



import hvplot.pandas

import holoviews as hv

import matplotlib.pyplot as plt



from sklearn.base import BaseEstimator, MetaEstimatorMixin

from sklearn.utils.metaestimators import if_delegate_has_method

from sklearn.datasets import load_files

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.cluster import DBSCAN



from umap import UMAP

from gensim.sklearn_api import D2VTransformer, W2VTransformer



hv.extension('bokeh')
! find /kaggle/working -type f -delete
# load texts

target_dir = Path('/kaggle/working/articles')

target_dir.mkdir()

r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))

table = str.maketrans(dict.fromkeys(string.punctuation.replace('-','')))



# look though the sources

for dirname, _, filenames in os.walk('/kaggle/input'):

    # look through the journals and preprint servers

    for filename in filenames:

        try:

            file_path = Path(dirname, filename)



            # extract the json for articles

            if str(file_path).endswith('.json'):

                with open(file_path, 'r') as f:

                    loaded_json = json.loads(f.read())



                # save articles names as folder titles

                title = loaded_json['metadata']['title']

                if title == '':

                    continue

                else:

                    name = '-'.join(re.split('(?<=[\.\,\?\!\(\)\:\/])\s*', title)[0].replace(" ", "-").split()[:5])

                    name = name.split('/')[0]



                    if len(name) > 100:

                        name = name[:100]



                    name = name.translate(table)



                    article_folder = target_dir / name



                    if article_folder.exists():

                        continue

                    else:

                        article_folder.mkdir()



                        # write out the paragraphs as documents

                        for i, paragraph in enumerate(loaded_json['body_text']):

                            if paragraph['text'] == '':

                                continue

                            else:

                                paragraph_file = article_folder / f'{i}.txt'

                                with open(paragraph_file, 'w') as f:

                                    f.write(paragraph['text'])

        except OSError as e:

            break
data = load_files('/kaggle/working/articles', encoding='utf-8')
! find /kaggle/working/articles -type f -delete
print(f'{len(data.data)} paragraphs in corpus')
class Pipeline_(Pipeline):

    def topics(self, X: np.ndarray, with_final=False) -> np.ndarray:

        """

        :param X: Feature space array

        :param with_final: This will run transform on all the steps

                +           of the pipeline but the last, defaults to False

        :type with_final: bool, optional

        :return: transformed pipeline (self.regressor)

        """

        Xt = X

        for _, _, transform in self._iter(with_final=with_final):

            Xt = transform.transform(Xt)

        return Xt
subsampling = 100

X = data.data[::subsampling]
pipeline = Pipeline_([('tfidf', CountVectorizer()),

                      ('lda', LatentDirichletAllocation(50)),

                      ('umap', UMAP(n_components=2, n_neighbors=10, metric='cosine'))])

pipeline.fit(X)

Z = pipeline.transform(X)

T = pipeline.topics(X)



L = DBSCAN().fit_predict(X=Z)
(pd.DataFrame(Z, columns=['Topic Component 1', 'Topic Component 2'])

 .assign(cluster = L)

 .sample(1000)

 .assign(cluster = lambda df: df.cluster.astype(str))

 .hvplot.scatter(x='Topic Component 1', y='Topic Component 2', c='cluster', title='CORD-19 Research Topic Clusters in Topic Embedding Space', legend=False))
(pd.DataFrame(Z, columns=['Topic Component 1', 'Topic Component 2'])

 .assign(article = np.array(data.target_names)[data.target].tolist()[::subsampling])

 .sample(1000)

 .assign(article = lambda df: df.article.astype(str))

 .hvplot.scatter(x='Topic Component 1', y='Topic Component 2', c='article', title='CORD-19 Research Articles in Topic Embedding Space', legend=False))
class Tokenizer(BaseEstimator, MetaEstimatorMixin):

    """Tokenize input strings based on a simple word-boundary pattern."""

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        ## split on word-boundary. A simple technique, yes, but mirrors what sklearn does to preprocess:

        ## https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/feature_extraction/text.py#L261-L266

        token_pattern = re.compile(r"(?u)\b\w\w+\b")

        parser = lambda doc: token_pattern.findall(doc)

        func = lambda df: df.apply(parser)

        return (pd.Series(X)

                .pipe(dd.from_pandas, npartitions=8)

                .map_partitions(func, meta=(None, 'object'))

                .compute(scheduler='processes')

                .values)
pipeline_d2v = Pipeline([

    ('tokenize', Tokenizer()),

    ('d2v', D2VTransformer(size=50, iter=100)),

    ('umap', UMAP(n_components=2, n_neighbors=15, metric='cosine', min_dist=0))

])



Z_d2v = pipeline_d2v.fit_transform(X)



L_d2v = DBSCAN().fit_predict(X=Z_d2v)
(pd.DataFrame(Z_d2v, columns=['Component 1', 'Component 2'])

 .assign(cluster = L_d2v)

 .sample(1000)

 .assign(cluster = lambda df: df.cluster.astype(str))

 .hvplot.scatter(x='Component 1', y='Component 2', c='cluster', title='CORD-19 Research Document Clusters in Document Embedding Space', legend=False))
(pd.DataFrame(Z_d2v, columns=['Component 1', 'Component 2'])

 .assign(article = np.array(data.target_names)[data.target].tolist()[::subsampling])

 .sample(1000)

 .assign(article = lambda df: df.article.astype(str))

 .hvplot.scatter(x='Component 1', y='Component 2', c='article', title='CORD-19 Research Articles in Document Embedding Space', legend=False))
class W2VTransformerDocLevel(W2VTransformer):

    """Extend Gensim's Word2Vec sklearn-wrapper class to further transform word-vectors into doc-vectors by

    averaging the words in each document."""

    

    def __init__(self, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3, seed=1,

                 workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,

                 trim_rule=None, sorted_vocab=1, batch_words=10000):

        super().__init__(size, alpha, window, min_count, max_vocab_size, sample, seed, workers, min_alpha, sg, hs, negative, cbow_mean, hashfxn, iter, null_word, trim_rule, sorted_vocab, batch_words)

    

    def transform(self, docs):      

        doc_vecs = []

        for doc in docs:

            ## for each document generate a word matrix

            word_vectors_per_doc = []

            for word in doc:

                ## handle out-of vocabulary words

                if word in self.gensim_model.wv:

                    word_vectors_per_doc.append(self.gensim_model.wv[word])



            word_vectors_per_doc = np.array(word_vectors_per_doc)

            ## take the column-wise mean of this matrix and store

            doc_vec = word_vectors_per_doc.mean(axis=0)

            

            if doc_vec.shape != (50,):

                warnings.warn('Empty vector')

                doc_vec = np.zeros((50,))

                

                

            doc_vecs.append(doc_vec)

        return np.stack(doc_vecs)
pipeline_w2v = Pipeline([

    ('tokenize', Tokenizer()),

    ('w2v', W2VTransformerDocLevel(size=50, iter=100)),

    ('umap', UMAP(n_components=2, n_neighbors=15, metric='cosine', min_dist=0))

])



Z_w2v = pipeline_w2v.fit_transform(X)



L_w2v = DBSCAN().fit_predict(X=Z_w2v)
(pd.DataFrame(Z_w2v, columns=['Component 1', 'Component 2'])

 .assign(cluster = L_w2v)

 .sample(1000)

 .assign(cluster = lambda df: df.cluster.astype(str))

 .hvplot.scatter(x='Component 1', y='Component 2', c='cluster', title='CORD-19 Research Clusters in Word-vector-level Document Embedding Space', legend=False))
(pd.DataFrame(Z_w2v, columns=['Component 1', 'Component 2'])

 .assign(article = np.array(data.target_names)[data.target].tolist()[::subsampling])

 .sample(1000)

 .assign(article = lambda df: df.article.astype(str))

 .hvplot.scatter(x='Component 1', y='Component 2', c='article', title='CORD-19 Research Articles in Word-vector-level Document Embedding Space', legend=False))