# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from scipy.stats import randint

import seaborn as sns # used for plot interactive graph. 

import matplotlib.pyplot as plt

import seaborn as sns

from io import StringIO

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import chi2

from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn import metrics

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import re

import re

import sys

import nltk

# !{sys.executable} -m spacy download en

import re, numpy as np, pandas as pd

from pprint import pprint



# Gensim

import gensim, spacy, logging, warnings

import gensim.corpora as corpora

from gensim.utils import lemmatize, simple_preprocess

from gensim.models import CoherenceModel

import matplotlib.pyplot as plt



# NLTK Stop words

from nltk.corpus import stopwords



import re

import re

import sys

import nltk

# !{sys.executable} -m spacy download en

import re, numpy as np, pandas as pd

from pprint import pprint



# Gensim

import gensim, spacy, logging, warnings

import gensim.corpora as corpora

from gensim.utils import lemmatize, simple_preprocess

from gensim.models import CoherenceModel

import matplotlib.pyplot as plt



# NLTK Stop words

from nltk.corpus import stopwords







import spacy #load spacy

nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])

stops = stopwords.words("english")

stops.extend(['please','from', 'list','with','subject', 'hi','re','hello','thankyou', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])







%matplotlib inline

warnings.filterwarnings("ignore",category=DeprecationWarning)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
data1 = pd.read_csv("/kaggle/input/train.csv")
###calculate tf idf feature vector from combined text column

data1["body"] = data1["name_of_drug"]+data1["use_case_for_drug"]+data1["review_by_patient"]

data1['body'] = data1['body'].astype(str)

data1['body'] = data1['body'].map(lambda x: x.lower())

data1['body'] = data1['body'].map(lambda x: re.sub(r'[^a-zA-Z0-9_\s]+', '', x))







def normalize(comment, lowercase, remove_stopwords):

   if lowercase:

       comment = comment.lower()

   comment = nlp(comment)

   lemmatized = list()

   for word in comment:

       lemma = word.lemma_.strip()

       if lemma:

           if not remove_stopwords or (remove_stopwords and lemma not in stops):

               lemmatized.append(lemma)

   return " ".join(lemmatized)





data1['body'] = data1['body'].apply(normalize, lowercase=True, remove_stopwords=True)



tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10,

                        ngram_range=(0, 1),

                        token_pattern='[a-zA-Z0-9]{3,}',

                        stop_words= stops)



# We transform each complaint into a vector

features = tfidf.fit_transform(data1.body).toarray()





print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams)" %(features.shape))
###derive new variables from text column 

import string

punctuation=string.punctuation

data1['word_count']=data1['body'].apply(lambda x: len(str(x).split(" ")))

data1['char_count'] = data1['body'].str.len()

def avg_word(sentence):

    words = sentence.split()

    return (sum(len(word) for word in words)/len(words))



data1['avg_word'] = data1['body'].apply(lambda x: avg_word(x))



data1['stopwords'] = data1['body'].apply(lambda x: len([x for x in x.split() if x in stops]))

data1['numerics'] = data1['body'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

data1['word_density'] = data1['char_count'] / (data1['word_count']+1)

data1['punctuation_count'] = data1['body'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation)))
import datetime

data1["year"] = pd.DatetimeIndex(data1["drug_approved_by_UIC"]).year

data1["month"] = pd.DatetimeIndex(data1["drug_approved_by_UIC"]).month

data1["day"] = pd.DatetimeIndex(data1["drug_approved_by_UIC"]).day
##taking up the tf idf features as derived columns

data2 = pd.DataFrame(features)
###creating final data by combining two data sets

data_final = pd.concat([data1,data2],axis =1)
import gc

gc.collect()
###data for final cross validation

## data has been divided into train and test

## please note that i m doing cross validation and accuracy checking on train data and test data will be used for validation

X = data_final.drop(['body','patient_id','base_score','name_of_drug','use_case_for_drug','review_by_patient','drug_approved_by_UIC'], axis = 1)  # Collection of documents

y = data_final['base_score'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.1,

                                                    random_state = 0)

import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)
from sklearn.metrics import mean_absolute_error
# "Learn" the mean from the training data

mean_train = np.mean(y_train)

# Get predictions on the test set

baseline_predictions = np.ones(y_test.shape) * mean_train

# Compute MAE

mae_baseline = mean_absolute_error(y_test, baseline_predictions)

print("Baseline MAE is {:.2f}".format(mae_baseline))
params = {

    # Parameters that we are going to tune.

    'max_depth':9,

    'min_child_weight': 5,

    'eta':.3,

    'subsample': 1,

    'colsample_bytree': 1,

    # Other parameters

    'objective':'reg:linear',

}
params['eval_metric'] = "mae"
num_boost_round = 100
model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=10

)

print("Best MAE: {:.2f} with {} rounds".format(

                 model.best_score,

                 model.best_iteration+1))
num_boost_round = model.best_iteration + 1

best_model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")]

)
mean_absolute_error(best_model.predict(dtest), y_test)
best_model.save_model("final_model.model")