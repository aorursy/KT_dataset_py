from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import FunctionTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline



import numpy as np

import os

import pickle

import numpy as np

import pandas as pd

from scipy.sparse import hstack



import eli5



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

import seaborn as sns

from IPython.display import display_html

from sklearn.base import BaseEstimator, TransformerMixin

import pdb
PATH_TO_DATA = '../input/'

SEED = 17
sites = ['site%s' % i for i in range(1, 11)]

times = ['time%s' % i for i in range(1, 11)]

path_to_train=os.path.join(PATH_TO_DATA, 'train_sessions.csv')

path_to_test=os.path.join(PATH_TO_DATA, 'test_sessions.csv')

path_to_site_dict=os.path.join(PATH_TO_DATA, 'site_dic.pkl')

train_df = pd.read_csv(path_to_train,

                   index_col='session_id', parse_dates=times)

test_df = pd.read_csv(path_to_test,

                  index_col='session_id', parse_dates=times)



with open(path_to_site_dict, 'rb') as f:

    site2id = pickle.load(f)

# create an inverse id _> site mapping

id2site = {v:k for (k, v) in site2id.items()}

# we treat site with id 0 as "unknown"

id2site[0] = 'unknown'
class Debug(BaseEstimator, TransformerMixin):



    def transform(self, X):

        print("Degugger Start")

        print(X[1:5])

        # what other output you want

        print("End")

        return X



    def fit(self, X, y=None, **fit_params):

        return self
def concatfunction(data):

    return data[sites].fillna(0).astype('int').apply(lambda row: 

                                                     ' '.join([id2site[i] for i in row]), axis=1).tolist()
class add_time_features(BaseEstimator, TransformerMixin):

    """Extract time features from datetime column"""



    def __init__(self,column='time1',add_hour=False):

        self.column=column

        self.add_hour=add_hour



    def transform(self, data, y=None):

        """The workhorse of this feature extractor"""

        times = ['time%s' % i for i in range(1, 11)]

        times=data[times]

        hour = times[self.column].apply(lambda ts: ts.hour)

        morning = ((hour >= 7) & (hour <= 11)).astype('int').values.reshape(-1, 1)

        day = ((hour >= 12) & (hour <= 18)).astype('int').values.reshape(-1, 1)

        evening = ((hour >= 19) & (hour <= 23)).astype('int').values.reshape(-1, 1)

        night = ((hour >= 0) & (hour <=6)).astype('int').values.reshape(-1, 1)

        objects_to_hstack = [ morning, day, evening, night]

        feature_names = ['morning', 'day', 'evening', 'night']

        if self.add_hour:

        # scale hour dividing by 24

            objects_to_hstack.append(hour.values.reshape(-1, 1) / 24)

            feature_names.append('hour')

        return pd.DataFrame(np.hstack(objects_to_hstack),columns=feature_names,index=data.index)



    def fit(self, data, y=None):

        """Returns `self` unless something different happens in train and test"""

        return self
vectorizer_params={'ngram_range': (1, 5), 

                   'max_features': 100000,

                   'tokenizer': lambda s: s.split()}

time_split = TimeSeriesSplit(n_splits=10)


# data --+-->concatenate sites in a session-->tf-idf vectorizer--+-->FeatureUnion-->Logistic Regression

#        |                                                       |

#        +--> extracting time features from start date column  --+

from sklearn.preprocessing import FunctionTransformer



pipeline = Pipeline( [

        ('union', FeatureUnion(

            transformer_list=[

                    ('tf-idf features',

                     Pipeline([  

                            ('concatenate sites',FunctionTransformer(concatfunction, validate=False)),  

                            ('vectorizing text',TfidfVectorizer(**vectorizer_params))

                             ])

                    )

                    ,

                    ('time features',

                     Pipeline([

                             ('time1 features',add_time_features(column='time1',add_hour=False))

                            ])

                    )

                            ]

                            )

        ),

        ('classifier',LogisticRegression(C=1, random_state=SEED, solver='liblinear'))

                    ])

pipeline.fit(train_df.drop(columns='target'),train_df['target'])
pipeline.predict(test_df)