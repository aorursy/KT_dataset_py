# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

import keras

import nltk.corpus

if '/usr/share/nltk_data' in nltk.data.path:

    nltk.data.path.remove('/usr/share/nltk_data')

nltk.data.path.append('../input/')

nltk.data.path
class BrownCorpus(object):

    def __init__(self):

        self.brown =  nltk.corpus.LazyCorpusLoader('brown', nltk.corpus.CategorizedTaggedCorpusReader, r'c[a-z]\d\d',

                                                    cat_file='cats.csv', tagset='brown', encoding="ascii",

                                                    nltk_data_subdir='brown-corpus/brown')

    def __iter__(self):

        for (tag,doc) in enumerate(self.brown.paras()):

            yield gensim.models.doc2vec.TaggedDocument(sum(doc,[]),[tag])

            

model = gensim.models.doc2vec.Doc2Vec(BrownCorpus(),

                                      dm_concat=1)
model.vector_size
def vectorize_cell(cell):

    return model.infer_vector(list(gensim.utils.tokenize(cell)))



def vectorize(data):

    return np.vstack([np.concatenate([vectorize_cell(row[cell])

                                      for cell in ('question_title','question_body','answer')])

                     for (i,row) in data.iterrows()])
training_data = pd.read_csv('../input/google-quest-challenge/train.csv',

                           index_col='qa_id')

training_data
training_vectors = vectorize(training_data)

training_vectors
predictor = keras.models.Sequential([keras.layers.Dense(100,

                                                       input_shape=(300,),

                                                       activation='softplus'),

                                     keras.layers.Dense(100,

                                                       activation='softplus'),

                                     keras.layers.Dense(30,

                                                       activation='sigmoid')])



predictor.compile(optimizer='nadam',

                  loss='mean_squared_error')

columns_to_predict = pd.read_csv('../input/google-quest-challenge/sample_submission.csv',

                                 index_col='qa_id').columns

predictor.fit(training_vectors,

              training_data.loc[:,columns_to_predict].values,

              epochs=100)
test_data = pd.read_csv('../input/google-quest-challenge/test.csv',

                        index_col='qa_id')

test_vectors = vectorize(test_data)

results = pd.DataFrame(predictor.predict(test_vectors),

                       index = test_data.index,

                       columns = columns_to_predict)

results
results.to_csv('submission.csv')