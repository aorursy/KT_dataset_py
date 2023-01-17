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
import nltk

nltk.download('stopwords')

import zipfile
import pandas as pd

import numpy as np

import xgboost as xgb

from tqdm import tqdm

from sklearn.svm import SVC

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping

import nltk

from nltk import word_tokenize

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
zf = zipfile.ZipFile('/kaggle/input/spooky-author-identification/train.zip') 

train = pd.read_csv(zf.open('train.csv'))

zf = zipfile.ZipFile('/kaggle/input/spooky-author-identification/test.zip') 

test = pd.read_csv(zf.open('test.csv'))

zf = zipfile.ZipFile('/kaggle/input/spooky-author-identification/sample_submission.zip') 

sample = pd.read_csv(zf.open('sample_submission.csv'))
train.head()
test.head()
train.shape,test.shape
label_encoder = preprocessing.LabelEncoder()

y = label_encoder.fit_transform(train['author'])
X_train,X_valid,y_train,y_valid = train_test_split(train['text'],y,test_size=0.2,random_state=42,shuffle=True,stratify=y)
X_train.shape,X_valid.shape
def multiclass_logloss(actual, predicted, eps=1e-15):

    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes

    :param predicted: Matrix with class predictions, one probability per class

    """

    # Convert 'actual' to a binary array if it's not already:

    if len(actual.shape) == 1:

        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))

        for i, val in enumerate(actual):

            actual2[i, val] = 1

        actual = actual2



    clip = np.clip(predicted, eps, 1 - eps)

    rows = actual.shape[0]

    vsota = np.sum(actual * np.log(clip))

    return -1.0 / rows * vsota
X_train
X_train = pd.DataFrame(X_train)

X_valid = pd.DataFrame(X_valid)
X_train.reset_index(inplace=True)

X_valid.reset_index(inplace=True)
sentences_train = [X_train.loc[i,'text'] for i in range(len(X_train['text']))]

sentences_test = [X_valid.loc[i,'text'] for i in range(len(X_valid['text']))]
len(sentences_train)
import re

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def clean_text(para):

    corpus = []



    for i in range(len(para)):

        review = re.sub('[^a-zA-Z]',' ',para[i])

        review = review.lower()

        review = review.split()

        review = [porter.stem(word) for word in review if word not in set(stopwords.words('english'))]

        review = ' '.join(review)

        corpus.append(review)



    return corpus 
corpus_train = clean_text(sentences_train)
corpus_test = clean_text(sentences_test)
tf = TfidfVectorizer(max_features=None,ngram_range=(1,3),analyzer='word')

X_train = tf.fit_transform(corpus_train).toarray()
X_valid = tf.transform(corpus_test).toarray()
X_train
X_valid
Lr = LogisticRegression()
Lr.fit(X_train,y_train)
ypred_Lr = Lr.predict_proba(X_valid)
multiclass_logloss(y_valid,ypred_Lr)
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train,y_train)

ypred_naive_bayes = naive_bayes.predict_proba(X_valid)
multiclass_logloss(y_valid,ypred_naive_bayes)