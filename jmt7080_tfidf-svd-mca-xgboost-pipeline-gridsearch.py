# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train_variants = pd.read_csv('../input/training_variants')

test_variants = pd.read_csv('../input/test_variants')

train_text = pd.read_csv('../input/training_text', sep="\|\|", engine='python', 

                         header=None, skiprows=1, names=["ID","Text"])

test_text = pd.read_csv('../input/test_text', sep="\|\|", engine='python', 

                        header=None, skiprows=1, names=["ID","Text"])



train_combined = pd.merge(train_variants, train_text, how='left', on='ID').fillna('')

test_combined = pd.merge(test_variants, test_text, how='left', on='ID').fillna('')
import numpy as np

import pandas as pd

from xgboost import XGBClassifier

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.feature_extraction import DictVectorizer

from sklearn.utils.extmath import randomized_svd

from sklearn.decomposition import TruncatedSVD, PCA

from sklearn.preprocessing import LabelEncoder

from scipy import sparse

from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import train_test_split

#import mca

import xgboost as xgb

from xgboost import XGBClassifier, plot_importance

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from sklearn.metrics import log_loss, accuracy_score
class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):

        self.key = key



    def fit(self, X, y=None):

        return self



    def transform(self, X): 

        return X[self.key]

    

class DummyMCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_factors):

        self.n_factors = n_factors



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        mca_obj = mca.MCA(pd.get_dummies(X), TOL=1e-2)

        factors = pd.DataFrame(mca_obj.fs_r(N=self.n_factors), columns= ['mca_factor' + str(x) for x in range(self.n_factors)])

        cosines = pd.DataFrame(mca_obj.fs_r(N=self.n_factors), columns= ['mca_cosine' + str(x) for x in range(self.n_factors)])

        mca_df = pd.concat([factors, cosines], axis =1)

        return mca_df



class GetDummies(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self



    def transform(self, X): 

        dummy_df = pd.get_dummies(X[self.key])

        return dummy_df



pipeline = Pipeline(steps=[

    ('union', FeatureUnion(

        transformer_list=[

            ('transform_text',

             Pipeline([

                 ('select_column', ColumnSelector(key='Text')),

                 ('tfidf', TfidfVectorizer(analyzer='word', max_df=0.9, stop_words='english', norm='l2',

                                           sublinear_tf=True, use_idf=True)),

                 ('svd', TruncatedSVD(algorithm='randomized', n_components=100)),

             ]),

             ),

            """

            ('transform_categories',

             Pipeline([

                 ('select_columns', ColumnSelector(key=['Gene', 'Variation'])),

                 ('dummy_mca', DummyMCA(n_factors=207)),

             ]),

             ),

             """

            ('transform_categories',

             Pipeline([

                 ('select_columns', ColumnSelector(key=['Gene', 'Variation'])),

                 ('get_dummy', GetDummies()),

                 #('svd', TruncatedSVD(algorithm='randomized', n_components=207)),

             ]),

             ),

        ],

    )),

    ('xgb', XGBClassifier(learning_rate=0.13,

                  n_estimators=55,

                  max_depth=5,

                  min_child_weight=8,

                  gamma=0,

                  subsample=0.9,

                  colsample_bytree=0.6,

                  objective='multi:softprob',

                  nthread=4,

                  scale_pos_weight=1,

                  reg_alpha=0.0001,

                  seed=7)),

])
features_train = train_combined[['Gene', 'Variation', 'Text']]

target_train = lbe.fit_transform(train_combined['Class'])



features_test = test_combined[['Gene', 'Variation', 'Text']]





pipeline.fit(features_train, target_train)

pred = pipeline.predict_proba(features_test)
submission_df = pd.DataFrame(pred, columns=['class_'+str(c+1) for c in range(9)])

submission_df['ID'] = test_combined['ID']

submission_df.head()