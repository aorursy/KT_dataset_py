import pandas as pd
import numpy as np
from operator import itemgetter
from gensim.parsing.preprocessing import preprocess_string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases

from xgboost import XGBRegressor

from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import QuantileTransformer, LabelBinarizer, StandardScaler, MinMaxScaler, RobustScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, explained_variance_score, r2_score
from sklearn.decomposition import NMF, LatentDirichletAllocation

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [8, 8]

from nltk.tag import pos_tag
from nltk import word_tokenize

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
f = "../input/train.csv"
# f = "train.csv"
fr = "../input/resources.csv"
train = pd.read_csv(f)
dfr = pd.read_csv(fr)
import multiprocessing

def _apply_df(args):
    df, func, num, kwargs = args
    return num, df.apply(func, **kwargs)

def apply_by_multiprocessing(df,func,**kwargs):
    workers=kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, i, kwargs) for i,d in enumerate(np.array_split(df, workers))])  
    pool.close()
    result=sorted(result,key=lambda x:x[0])
    return pd.concat([i[1] for i in result])

def tokenize(x):
    return word_tokenize(x)

def count_punctuation(tokens, punctuation_char):
             return len([token for token in tokens if token == punctuation_char])
  
def preprocess_df(df, workers):
    if __name__ == "__main__":
        dfr = pd.read_csv(fr)
        dfr['total'] = dfr['price'] * dfr['quantity']
        dfr['has_zero'] = dfr['price'].apply(lambda x: 1 if x==0 else 0)
        dfr = dfr.groupby('id').agg('sum').reset_index()

        # merging essays
        df['student_description']=df['project_essay_1']
        df.loc[df.project_essay_3.notnull(),'student_description']=df.loc[df.project_essay_3.notnull(),'project_essay_1']+df.loc[df.project_essay_3.notnull(),'project_essay_2']
        df['project_description']=df['project_essay_2']

        df.loc[df.project_essay_3.notnull(),'project_description']=df.loc[df.project_essay_3.notnull(),'project_essay_3']+df.loc[df.project_essay_3.notnull(),'project_essay_4']

        df['project_subject_categories'] = df['project_subject_categories'].apply(lambda x: x.split(", "))
        df['project_subject_subcategories'] = df['project_subject_subcategories'].apply(lambda x: x.split(", "))
        df['teacher_prefix'] = df['teacher_prefix'].fillna('None')
        df = df.merge(dfr, how='inner', on='id')


        df['student_tokens'] = apply_by_multiprocessing(df['student_description'], tokenize, workers=workers)
        df['student_word_count'] = df['student_tokens'].apply(lambda x: len(x))
        df['student_unique_words'] = df['student_tokens'].apply(lambda x: len(set(x)))
        df['student_n_periods'] = df['student_tokens'].apply(lambda x: count_punctuation(x, '.'))
        df['student_n_commas'] = df['student_tokens'].apply(lambda x: count_punctuation(x, ','))
        df['student_n_questions'] = df['student_tokens'].apply(lambda x: count_punctuation(x, '?'))
        df['student_n_exclamations'] = df['student_tokens'].apply(lambda x: count_punctuation(x, '!'))
        df['student_word_len'] = df['student_tokens'].apply(lambda x: np.mean([len(token) for token in x]))
        
        del(df['student_tokens'])
    
        df['project_tokens'] = apply_by_multiprocessing(df['project_description'], tokenize, workers=workers)
        df['project_word_count'] = df['project_tokens'].apply(lambda x: len(x))
        df['project_unique_words'] = df['project_tokens'].apply(lambda x: len(set(x)))

        
        
        df['project_n_periods'] = df['project_tokens'].apply(lambda x: count_punctuation(x, '.'))
        df['project_n_commas'] = df['project_tokens'].apply(lambda x: count_punctuation(x, ','))
        df['project_n_questions'] = df['project_tokens'].apply(lambda x: count_punctuation(x, '?'))
        df['project_n_exclamations'] = df['project_tokens'].apply(lambda x: count_punctuation(x, '!'))
        df['project_word_len'] = df['project_tokens'].apply(lambda x: np.mean([len(token) for token in x]))
        del(df['project_tokens'])
        del(df['project_essay_1'])
        del(df['project_essay_2'])
        del(df['project_essay_3'])
        del(df['project_essay_4'])
        return df
%%time
train = preprocess_df(train, 32)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases

def read_corpus(df, tokens_only=False):
    for i, row in df.iterrows():
        tag = row.project_is_approved
        docs = [row.student_description, row.project_description]
        docs = [preprocess_string(doc) for doc in docs]
        if tokens_only:
            for doc in docs:
                yield doc
        else:
            for doc in docs:
                yield TaggedDocument(doc, [tag])
                # For training data, add tags

class DocStreamer(object):
    def __init__(self, df):
        self.df = df
    
    def __iter__(self):
        for i, row in self.df.iterrows():
            tag = row.project_is_approved
            docs = [row.student_description, row.project_description]
            docs = [preprocess_string(doc) for doc in docs]
            for doc in docs:
                yield TaggedDocument(doc, ["{}-{}".format(tag, i)])

%%time

# Better results when tags are equal?
not_approved = train.loc[train['project_is_approved']==0]
n_na = len(not_approved)
approved = train.loc[train['project_is_approved']==1]
approved = approved.sample(n_na)

docs_df = pd.concat([approved, not_approved], axis=0, copy=True)

# Shuffles the dataframe
docs = DocStreamer(docs_df.sample(frac=1))
doc_model = Doc2Vec(docs, vector_size=200, window=5, min_count=5, workers=32, epochs=1)
del docs
del not_approved
del approved
def doc2vec_classify(text, model, binary=False):
    doc = preprocess_string(text)
    inf_vector = model.infer_vector(doc)
    doc_sims = model.docvecs.most_similar([inf_vector])
    if binary:
        # Return most similar class, i.e. 0, 1
        doc_sims = sorted(doc_sims, key=itemgetter(1), reverse=True)
        return int(doc_sims[0][0].split("-")[0])
    else:
        # Return similarity to project_is_accepted
        return len([tag for tag, score in doc_model.docvecs.most_similar([inf_vector]) if tag.startswith('1')]) / 10
%%time
if __name__ == "__main__":
    train['student_description_sim'] = apply_by_multiprocessing(train['student_description'], doc2vec_classify, model=doc_model, binary=False, workers=16)
    train['project_description_sim'] = apply_by_multiprocessing(train['student_description'], doc2vec_classify, model=doc_model, binary=False, workers=16)
%%time
raw_X = train
raw_y = train['project_is_approved'].values

X_train, X_test, y_train, y_test = train_test_split(raw_X, raw_y, test_size=0.2)

mapper = DataFrameMapper([
    (['teacher_number_of_previously_posted_projects'], StandardScaler()),
    (['student_word_count'], StandardScaler()),
    (['project_word_count'], StandardScaler()),
    (['student_n_periods'], StandardScaler()),
    (['student_n_commas'], StandardScaler()),
    (['student_n_questions'], StandardScaler()),
    (['student_n_exclamations'], StandardScaler()),
    (['project_n_periods'], StandardScaler()),
    (['project_n_commas'], StandardScaler()),
    (['project_n_questions'], StandardScaler()),
    (['project_n_exclamations'], StandardScaler()),
    (['total'], StandardScaler()),
    (['quantity'], StandardScaler()),
    ('student_description', [TfidfVectorizer(use_idf=True, ngram_range=(1,2), stop_words='english', max_features=10000),
                                                      NMF(n_components=20)]),
    ('project_description', [TfidfVectorizer(use_idf=True, ngram_range=(1,2), stop_words='english', max_features=10000),
                                                      NMF(n_components=20)]),
    (['student_description_sim'], StandardScaler()),
    (['project_description_sim'], StandardScaler()),
], sparse=True)

X_train = mapper.fit_transform(X_train)
X_test = mapper.transform(X_test)
from imblearn.under_sampling import NeighbourhoodCleaningRule
sampler = NeighbourhoodCleaningRule(ratio='majority', n_jobs=32)
# sampler = RandomUnderSampler()

from collections import Counter
Counter(y_train)
%%time
# X_trainR, y_trainR = sampler.fit_sample(X_train, y_train)
X_trainR, y_trainR = X_train, y_train
# The scaled data
Counter(y_trainR)
import xgboost as xgb

xgb_params = {'eta': 0.001, 
                  'max_depth': 8,
                  'max_delta_step': 6,
                  'subsample': 0.8, 
                  'colsample_bytree': 0.8, 
                  'objective': 'binary:logistic', 
                  'eval_metric': 'auc'
                  }

d_train = xgb.DMatrix(X_trainR, y_trainR)
d_test = xgb.DMatrix(X_test, y_test)
watchlist = [(d_train, 'train'), (d_test, 'valid')]
model_xgb = xgb.train(xgb_params, d_train, 500, watchlist, verbose_eval=50, early_stopping_rounds=25)
test = pd.read_csv("../input/test.csv")
test = preprocess_df(test)
if __name__ == "__main__":
    test['project_description_sim'] = apply_by_multiprocessing(test['project_description'], doc2vec_classify, model=doc_model, binary=False, workers=32)
if __name__ == "__main__":
    test['project_description_bin'] = apply_by_multiprocessing(test['project_description'], doc2vec_classify, model=doc_model, binary=True, workers=32)
if __name__ == "__main__":
    test['student_description_sim'] = apply_by_multiprocessing(test['student_description'], doc2vec_classify, model=doc_model, binary=False, workers=32)    
if __name__ == "__main__":
    test['student_description_bin'] = apply_by_multiprocessing(test['student_description'], doc2vec_classify, model=doc_model, binary=True, workers=32)
X_test_actual = mapper.transform(test)
y_pred_actual = model_xgb.predict(X_test_actual)
my_submission = pd.DataFrame({'id': df_test.id, 'project_is_approved': y_pred_actual})
my_submission.to_csv('submission.csv', index=False)
df_test = pd.read_csv("../input/test.csv")
df_test = preprocess_df(df_test, 32)

X_test_actual = mapper.transform(df_test)
y_pred_actual = random_search.predict(X_test_actual)

my_submission = pd.DataFrame({'id': df_test.id, 'project_is_approved': y_pred_actual})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)