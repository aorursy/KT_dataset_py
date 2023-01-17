# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import gc

gc.enable()



import logging

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

import os

import numpy as np

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import catboost

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.gridspec as gridspec

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn import metrics



from sklearn.svm import SVC

import time

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
%%time

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')



train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(train.shape)

print(test.shape)
df_ltrain = train['P_emaildomain'] ## Taking only one categorical column sample . We can clean up this code by using smart loops 

df_ltest =  test ['P_emaildomain']## Taking only one categorical column sample . We can clean up this code by using smart loops 
df_ltrain.fillna('X',inplace = True) ## Filling the NaN with some value else the next operations will give error

df_ltest.fillna('X',inplace = True)
cv = CountVectorizer(max_features=10000, min_df = 0.1, max_df = 0.8) ###This just breaks

                                                                     ### the documents into various tokens 

sparse_train = cv.fit_transform(df_ltrain)

sparse_test  = cv.transform(df_ltest)
#sparse_data_train =  sparse.vstack([sparse_train, sparse_test]) ## original implementation had their train and test data together 
#Transform our sparse_data to corpus for gensim

corpus_data_gensim = gensim.matutils.Sparse2Corpus(sparse_train, documents_columns=False)
#Create dictionary for LDA model

vocabulary_gensim = {}

for key, val in cv.vocabulary_.items():

    vocabulary_gensim[val] = key

    

dict = Dictionary()

dict.merge_with(vocabulary_gensim)
print(vocabulary_gensim)
%%time

lda = LdaModel(corpus_data_gensim, num_topics = 5 ) ## Here we are creating 5 new features, so topic is 5.

def document_to_lda_features(lda_model, document):

    topic_importances = lda.get_document_topics(document, minimum_probability=0)

    topic_importances = np.array(topic_importances)

    return topic_importances[:,1]



lda_features = list(map(lambda doc:document_to_lda_features(lda, doc),corpus_data_gensim))
data_pd_lda_features = pd.DataFrame(lda_features)

data_pd_lda_features.head()
data_pd_lda_features.shape
## lets find out the correlation between newly created features

fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(20.7, 8.27)

sns.heatmap(data_pd_lda_features.corr(method = 'spearman'), cmap="coolwarm", ax = ax)