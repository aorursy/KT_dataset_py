# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')
train
train_data=train[['question1','question2','is_duplicate']]
train_data.shape
train_data.columns
train_data=train_data.dropna()
train_data.isna().value_counts()
train_data.shape
train_data=train_data.iloc[0:200,0:200]
train_data.shape
train_data
X=train_data[['question1','question2']]
y=train_data[['is_duplicate']]
X.shape
! pip install tpot
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from tpot import TPOTClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tpot.config import classifier_config_dict_light 
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
import copy
class IdentityTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X):
        return X

class TfidfTransformer(TransformerMixin):
    def __init__(self, text_columns, keep_columns=[], **kwargs):
        self.text_columns = text_columns if type(text_columns) is list else [text_columns]
        self.keep_columns = keep_columns if type(keep_columns) is list else [keep_columns]
        
        column_list = []
        for idx, text in enumerate(self.text_columns):
            column_list.append(('text' + str(idx), TfidfVectorizer(**kwargs), text))
        
        if len(keep_columns) > 0:
            column_list.append(('other', IdentityTransformer(), self.keep_columns))
        
        self.column_transformer = ColumnTransformer(column_list)
    def fit(self, X, y=None):
        self.column_transformer.fit(X, y)
        return self
    def transform(self, X):
        return self.column_transformer.transform(X) 
# using TPOT config
config = copy.deepcopy(classifier_config_dict_light)
config["__main__.TfidfTransformer"] = {
        "text_columns": [["question1","question2"]]
    }
tpot = TPOTClassifier(config_dict=config,verbosity=2, generations=5, population_size=2, early_stop=2, max_time_mins=2,
                     template='TfidfTransformer-Selector-Transformer-Classifier')
tpot.fit(X, y)
