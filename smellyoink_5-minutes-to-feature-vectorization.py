import time

import numpy as np

import pandas as pd

import gc
def split_cat(text):

    try:

        return text.split("/")

    except:

        return ("No Label", "No Label", "No Label")
%%time



start_time = time.time()

    



train = pd.read_table('../input/train.tsv', engine='c', 

                      dtype={'item_condition_id': 'category',

                             'shipping': 'category',

                            }, 

                     converters={'category_name': split_cat})

test = pd.read_table('../input/test.tsv', engine='c', 

                      dtype={'item_condition_id': 'category',

                             'shipping': 'category',

                            },

                    converters={'category_name': split_cat})

print('[{}] Finished load data'.format(time.time() - start_time))



nrow_test = train.shape[0]

dftt = train[(train.price < 1.0)]

train = train.drop(train[(train.price < 1.0)].index)

del dftt['price']

nrow_train = train.shape[0]

y = np.log1p(train['price'])

merge = pd.concat([train, dftt, test])

submission: pd.DataFrame = test[['test_id']]



del train, test

gc.collect()



merge['gencat_name'] = merge['category_name'].str.get(0).replace('', 'missing').astype('category')

merge['subcat1_name'] = merge['category_name'].str.get(1).fillna('missing').astype('category')

merge['subcat2_name'] = merge['category_name'].str.get(2).fillna('missing').astype('category')

merge.drop('category_name', axis=1, inplace=True)

print('[{}] Split categories completed.'.format(time.time() - start_time))



merge['item_condition_id'] = merge['item_condition_id'].cat.add_categories(['missing']).fillna('missing')

merge['shipping'] = merge['shipping'].cat.add_categories(['missing']).fillna('missing')

merge['item_description'].fillna('missing', inplace=True)

merge['brand_name'] = merge['brand_name'].fillna('missing').astype('category')

print('[{}] Handle missing completed.'.format(time.time() - start_time))
from scipy import sparse as sp

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from multiprocessing import Pool

from collections import Counter

from contextlib import closing

import array

import dill

dill.settings['byref'] = True

from operator import itemgetter

from numbers import Integral

from six import string_types

from bisect import bisect_left, bisect_right



def apply_packed_function_for_map(x,):

    """

    https://stackoverflow.com/questions/8804830/python-multiprocessing-pickling-error

    Unpack dumped function as target function and call it with arguments.



    :param (dumped_function, item, args, kwargs):

        a tuple of dumped function and its arguments

    :return:

        result of target function

    """

    dumped_function, item = x

    target_function = dill.loads(dumped_function)

    res = target_function(item)

    return res





def pack_function_for_map(target_function, items):

    """

    https://stackoverflow.com/questions/8804830/python-multiprocessing-pickling-error

    Pack function and arguments to object that can be sent from one

    multiprocessing.Process to another. The main problem is:

        «multiprocessing.Pool.map*» or «apply*»

        cannot use class methods or closures.

    It solves this problem with «dill».

    It works with target function as argument, dumps it («with dill»)

    and returns dumped function with arguments of target function.

    For more performance we dump only target function itself

    and don't dump its arguments.

    How to use (pseudo-code):



        ~>>> import multiprocessing

        ~>>> images = [...]

        ~>>> pool = multiprocessing.Pool(100500)

        ~>>> features = pool.map(

        ~...     *pack_function_for_map(

        ~...         super(Extractor, self).extract_features,

        ~...         images,

        ~...         type='png'

        ~...         **options,

        ~...     )

        ~... )

        ~>>>



    :param target_function:

        function, that you want to execute like  target_function(item, *args, **kwargs).

    :param items:

        list of items for map

    :param args:

        positional arguments for target_function(item, *args, **kwargs)

    :param kwargs:

        named arguments for target_function(item, *args, **kwargs)

    :return: tuple(function_wrapper, dumped_items)

        It returs a tuple with

            * function wrapper, that unpack and call target function;

            * list of packed target function and its' arguments.

    """

    dumped_function = dill.dumps(target_function)

    dumped_items = list(zip([dumped_function] * len(items), items))

#     print('done pickling')

    return apply_packed_function_for_map, dumped_items



class MPCountVectorizer(CountVectorizer):

    """Subclass CountVectorizer and make multiprocessing """

    def __init__(self, n_jobs, chunk_size, save_vocab, input='content', encoding='utf-8',

                 decode_error='strict', strip_accents=None,

                 lowercase=True, preprocessor=None, tokenizer=None,

                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",

                 ngram_range=(1, 1), analyzer='word',

                 max_df=1.0, min_df=1, max_features=None,

                 vocabulary=None, binary=False, dtype=np.int64):



        self.n_jobs = n_jobs

        self.chunk_size = chunk_size

        self.save_vocab = save_vocab

        

        super().__init__(input=input, encoding=encoding,

                 decode_error=decode_error, strip_accents=strip_accents,

                 lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer,

                 stop_words=stop_words, token_pattern=token_pattern,

                 ngram_range=ngram_range, analyzer=analyzer,

                 max_df=max_df, min_df=min_df, max_features=max_features,

                 vocabulary=vocabulary, binary=binary, dtype=dtype)

        

    def _task_multiprocess(self, task, args, vocabulary, analyzer):

        def init(x, y):

            global vocabulary_, analyzer_

            vocabulary_ = x

            analyzer_ = y

            

        with closing(Pool(self.n_jobs, 

                          initializer=init, 

                          initargs=(vocabulary, analyzer),

                          maxtasksperchild=2)) as pool:

            results = pool.map_async(*pack_function_for_map(task, args))

            results.wait(timeout=600)

            if results.ready():  

                results = results.get()

        return results

    

    def _chunker(self, l, n):

            """Yield successive n-sized chunks from l."""

            for i in range(0, len(l), n):

                yield l.iloc[i:i + n]

    

    def _feat_vector_task(self, raw_documents):

        """ Subprocess to create sparse feature matrix

        """

        

        j_indices = []

        values = array.array("i")

        indptr = array.array("i")

        indptr.append(0)

        

        for doc in raw_documents:

            feature_counter = {}

            for feature in analyzer_(doc):

                try:

                    feature_idx = vocabulary_[feature]

                    if feature not in feature_counter:

                        feature_counter[feature_idx] = 1

                    else:

                        feature_counter[feature_idx] += 1

                except KeyError:

                    continue

            j_indices += feature_counter.keys()

            values.extend(feature_counter.values())

            indptr.append(len(j_indices))

        

        return sp.csr_matrix((values, j_indices, indptr),

                             shape=(len(indptr) - 1, len(vocabulary_)),

                             dtype=self.dtype)



    def _word_count_task(self, raw_documents):



        word_counts = {}

        for doc in raw_documents:

            for tok in analyzer_(doc):

                if tok in word_counts:

                    word_counts[tok] += 1

                else:

                    word_counts[tok] = 1



        return word_counts

    

    def _count_vocab(self, raw_documents, fixed_vocab):

        """Create sparse feature matrix, and vocabulary where fixed_vocab=False

        """



#         print('create chunks')

        chunks = list(self._chunker(raw_documents, self.chunk_size))

        analyzer = self.build_analyzer()



        if fixed_vocab:

            vocabulary = self.vocabulary_



        else:

            max_df = self.max_df

            min_df = self.min_df



#             print('build vocabulary')

            self.vocabulary_ = None

            

            vocabulary = {}

            partial_counts = self._task_multiprocess(

                self._word_count_task,

                chunks,

                None,

                analyzer

            )

#             print('merge vocabulary')

            word_counts = Counter(partial_counts[0])

            for count in partial_counts[1:]:

                word_counts.update(count)

            word_counts = dict(word_counts)

        

#             print('filter vocabulary')

            n_doc = len(raw_documents)

            max_features = self.max_features

            max_doc_count = (max_df

                                 if isinstance(max_df, Integral)

                                 else max_df * n_doc)

            min_doc_count = (min_df

                             if isinstance(min_df, Integral)

                             else min_df * n_doc)



            if max_doc_count < min_doc_count:

                raise ValueError("max_df corresponds to < documents than min_df")



#             removed_terms = []

            word_counts = sorted(word_counts.items(), key=itemgetter(1))

            keys, vals = list(zip(*word_counts))

            left_index = bisect_left(vals, min_doc_count)

            right_index = bisect_right(vals, max_doc_count)

            

#             removed_terms += keys[:left_index]

#             removed_terms += keys[right_index:]

            word_counts = word_counts[left_index:right_index]

            if max_features:

#                 removed_terms += keys[:-max_features]

                word_counts = word_counts[-max_features:]

            vocabulary = dict(zip([kv[0] for kv in word_counts], 

                                   range(len(word_counts))))

                     

#             self.stop_words_ = removed_terms



            if not vocabulary:

                raise ValueError("empty vocabulary; perhaps the documents only"

                                 " contain stop words or min_df, max_df too stringent")

        

#         self.vocabulary_ = word_counts

        

#         print('create counts')

        partial_X = self._task_multiprocess(

            self._feat_vector_task,

            chunks,

            vocabulary,

            analyzer

        )

                        

#         print('merge counts')

        X = sp.vstack(partial_X)

        X.sort_indices()

        

#         print('done')

        

        return word_counts, X

    

    def fit_transform(self, raw_documents, y=None):

        """Learn the vocabulary dictionary and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently

        implemented.

        Parameters

        ----------

        raw_documents : iterable

            An iterable which yields either str, unicode or file objects.

        Returns

        -------

        X : array, [n_samples, n_features]

            Document-term matrix.

        """

        # We intentionally don't call the transform method to make

        # fit_transform overridable without unwanted side effects in

        # TfidfVectorizer.

        if isinstance(raw_documents, string_types):

            raise ValueError(

                "Iterable over raw text documents expected, "

                "string object received.")



        self._validate_vocabulary()

        max_df = self.max_df

        min_df = self.min_df

        max_features = self.max_features



        vocabulary, X = self._count_vocab(raw_documents,

                                          self.fixed_vocabulary_)



        if self.binary:

            X.data.fill(1)

            

        if not self.fixed_vocabulary_ and self.save_vocab:

            self.vocabulary_ = vocabulary



        return X

from sklearn.base import BaseEstimator, TransformerMixin



class ItemSelector(BaseEstimator, TransformerMixin):



    def __init__(self, field, dtype=None):

        self.field = field

        self.dtype = dtype



    def fit(self, x, y=None):

        return self



    def transform(self, dataframe):

        if self.dtype == 'category':

            return dataframe[self.field].cat.codes[:, None]

        elif self.dtype == 'numeric':

            return dataframe[self.field][:, None]

        else:

            return dataframe[self.field]
%%time

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.preprocessing import OneHotEncoder



vectorizer_cat = FeatureUnion([

    ('item_condition_id', Pipeline([

        ('selector', ItemSelector(field='item_condition_id', dtype='category')),

        ('ohe', OneHotEncoder())

    ])),

    ('shipping', Pipeline([

        ('selector', ItemSelector(field='shipping', dtype='category')),

        ('ohe', OneHotEncoder())

    ])),

    ('gencat_name', Pipeline([

        ('selector', ItemSelector(field='gencat_name', dtype='category')),

        ('ohe', OneHotEncoder())

    ])),

    ('subcat1_name', Pipeline([

        ('selector', ItemSelector(field='subcat1_name', dtype='category')),

        ('ohe', OneHotEncoder())

    ])),

    ('subcat2_name', Pipeline([

        ('selector', ItemSelector(field='subcat2_name', dtype='category')),

        ('ohe', OneHotEncoder())

    ])),

    ('brand_name', Pipeline([

        ('selector', ItemSelector(field='brand_name', dtype='category')),

        ('ohe', OneHotEncoder())

    ])),

    ('name', Pipeline([

        ('selector', ItemSelector(field='name')),

        ('cv', MPCountVectorizer(

            n_jobs=4,

            chunk_size=50000,

            save_vocab=False,

            ngram_range=(1, 2),

            stop_words='english',

            min_df=10

        )),

    ])),

    ('item_description', Pipeline([

        ('selector', ItemSelector(field='item_description')),

        ('cv', MPCountVectorizer(

            n_jobs=4,

            chunk_size=50000,

            save_vocab=False,

            ngram_range=(1, 3),

            stop_words='english',

            max_features=1500000

        )),

        ('tfidf', TfidfTransformer()

        )

    ]))

], n_jobs=1)



X_train = vectorizer_cat.fit_transform(merge)



print('[{}] Data vectorization completed'.format(time.time() - start_time))
print('Total examples: {}, total features: {}'.format(X_train.shape[0], X_train.shape[1]))
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

train_X, valid_X, train_y, valid_y = train_test_split(X_train[:nrow_train], y, test_size = 0.1, random_state = 144)



model = Ridge(alpha=.5, copy_X=True, fit_intercept=True, max_iter=100,

      normalize=False, random_state=101, solver='auto', tol=0.01)

model.fit(train_X, train_y)

print('[{}] Train ridge completed'.format(time.time() - start_time))



# valid_X = lsa.transform(valid_X)

predsR = model.predict(valid_X)

print('[{}] Predict ridge completed'.format(time.time() - start_time))



def rmsle(y, y0):

    assert len(y) == len(y0)

    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

print('valid rmsle is', rmsle(np.expm1(predsR), np.expm1(valid_y)))