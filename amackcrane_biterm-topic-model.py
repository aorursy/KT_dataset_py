!pip install biterm
from biterm.cbtm import oBTM

import biterm.utility

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer

from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.decomposition import PCA



import numpy as np

import pandas as pd

import functools

import math



import matplotlib.pyplot as plt

%matplotlib inline
class BitermModel(BaseEstimator, TransformerMixin):

    """in: array of string docs

    out: doc-topic matrix"""

    def __init__(self, full=False, n_topics=20, it=50, chunk=100,

                max_df=.1, min_df=2, strip_accents='unicode'):

        # Vectorizer

        self.max_df = max_df

        self.min_df = min_df

        self.strip_accents = strip_accents

        

        # BTM

        self.full = full # False, or float \in (0,1]

        self.chunk = chunk

        self.it = it

        self.n_topics = n_topics



    def fit(self, X, y=None):

        # Vectorize

        self.vectorizer = CountVectorizer(max_df=self.max_df, min_df=self.min_df,

                                          strip_accents=self.strip_accents)

        # get document-word matrix

        self.doc_word = self.vectorizer.fit_transform(X)

        self.doc_word = self.doc_word.toarray()

        # translate doc-word matrix to biterms?

        biterms = biterm.utility.vec_to_biterms(self.doc_word)

        vocab = self.vectorizer.get_feature_names()



        # initialize biterm topic model

        self.btm = oBTM(num_topics=self.n_topics, V=vocab)

        

        # train online, or on a single chunk for quickness

        if self.full:

            end = round(self.full * len(biterms))

            print(str(math.ceil(float(end) / self.chunk)) + " iterations:")

            for i in range(0, end, self.chunk):

                fit_biterms = biterms[i:i+self.chunk]

                self.btm.fit(fit_biterms, iterations=self.it)

        else:

            indices = range(len(biterms))

            biterm_fit_ind = np.random.choice(indices, size=self.chunk, replace=False)

            fit_biterms = [biterms[i] for i in biterm_fit_ind]

            self.btm.fit(fit_biterms, iterations=self.it)

            

        return self



    def transform(self, X):

        # don't save doc_word this time -- it should correspond to the fitting...

        doc_word = self.vectorizer.transform(X)

        doc_word = doc_word.toarray()

        biterms = biterm.utility.vec_to_biterms(doc_word)

        

        self.doc_topic = self.btm.transform(biterms)

        # set 'nan' docs to zero for now

        self.doc_topic = np.where(np.isnan(self.doc_topic), 0, self.doc_topic)

        return self.doc_topic
# baseline parameters

vectorizer_args = {'topicmodel__max_df': .1, 'topicmodel__min_df': 2,

                    'topicmodel__strip_accents': 'unicode'}

# topicmodel__full should be False or a float in (0,1]

topic_args = {'topicmodel__it': 50, 'topicmodel__full': 1,

              'topicmodel__n_topics': 100}

baseline_args = dict(vectorizer_args, **topic_args)



# GridSearch parameters

param_grid = [{'topicmodel__it': [10, 20, 50]},

              {'topicmodel__full': [.05, .1, .3]},

              {'topicmodel__n_topics': [10, 20, 50]}]

gridsearch_params = {'refit': True, 'cv': 3, 'verbose': 0}



# quick formatting test

param_grid = {'topicmodel__it': [10], 'topicmodel__full': [.05],

             'topicmodel__n_topics': [10]}

pipe = Pipeline([('topicmodel', BitermModel()),

                 ('classifier', KNeighborsClassifier())], verbose=False)

pipe.set_params(**baseline_args)
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

# shuffle in case we train on a non-random subset

train = train.sample(frac=1, replace=False)

train.shape
train.head(5)
train['length'] = list(map(lambda x: len(x.split()), train.text))

# print(train.length.describe())

# print(train.target.astype('category').describe())

# print(train.keyword.astype('category').describe())

# print(train.keyword.value_counts()[:20])
labels = np.array(train.target)

labels.shape

check = BitermModel(full=.01, it=1).fit(train.text)

doc_word = check.doc_word

frequencies = np.add.reduce(doc_word, axis=0)

plt.hist(frequencies, bins=20, log=True)

#plt.title("Distribution of Word Frequencies")

pass
if isinstance(param_grid, list):

    nrows = len(param_grid)

else:

    nrows=1

    param_grid = [param_grid]



best_params = []

results = []

for i in range(nrows):

    gridsearch = GridSearchCV(pipe, param_grid=param_grid[i],

                             **gridsearch_params)

    gridsearch.fit(X=train.text, y=labels)

    results.append(pd.DataFrame(gridsearch.cv_results_))

    best_params.append(gridsearch.best_params_)



    # reset

    pipe.set_params(**baseline_args)
#for res in results:

#    print(res.filter(items=['params', 'mean_fit_time', 

#                            'mean_test_score']))
for i,res in enumerate(results):

    # pull out params_ columns...

    params = res.filter(like='param_')



    # how many params are there?

    cols = params.shape[1]



    if cols == 1:

        fig, ax = plt.subplots()

        xlims = [params.values.min(), params.values.max()]

        ylims = [res.mean_test_score.values.min(), 

                res.mean_test_score.values.max()]

        ax.plot(params, res.mean_test_score, 'b-', label="Test Score")

        ax.set_xlim(xlims)

        ax.set_ylim(ylims)

        ax.set_xlabel(params.columns[0])



        ax2 = ax.twinx()

        ax2.plot(params, res.mean_fit_time, 'r-', label="Fit Time")



        ax.set_ylabel("Score")

        ax2.set_ylabel("Time (s)")

        fig.legend()

    else:

        table = res.pivot_table(columns=params.columns[0], 

                              index=params.columns[1],

                              values=['mean_test_score', 'mean_fit_time',

                                     'rank_test_score'])

        print(table)

        #axs[i].axis('off')

        #axs[i].table(cellText=table.values, rowLabels=table.index, 

        #             colLabels=table.columns, loc=9)



new_params = functools.reduce(lambda x,y: dict(x, **y), best_params, {})

pipe.set_params(**new_params)

pipe.fit(X=train.text, y=labels)

pass
doc_topic = pipe['topicmodel'].doc_topic

pca = PCA(n_components=2)

doc_pc = pca.fit_transform(doc_topic)

doc_pc.shape

plt.scatter(x=doc_pc[:,0], y=doc_pc[:,1], c=labels)

pass
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test.head(5)
print(test.shape)

test_labels = pipe.predict(test.text)

print(test_labels.shape)
submission = pd.DataFrame({'id': test.id, 'target': test_labels})

submission.to_csv('submission.csv', index=False)

submission