# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

from nltk import word_tokenize

from nltk.corpus import stopwords

stop_words = stopwords.words('english')







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/amazon/train.csv')

train.shape
train.head(3)
train['topic'].nunique()
train['topic'].value_counts()
lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(train.topic.values)
# Remove white space

train['Review Text']=train['Review Text'].str.replace('\n', ' ',regex=True, case = False).str.strip()

train['Review Text']=train['Review Text'].str.replace('\s+', ' ',regex=True, case = False)

train['Review Text']=train['Review Text'].str.replace('/', ' ',regex=True, case = False).str.strip()

# lower case everything

train['Review Text']=train['Review Text'].str.lower()

#Remove stopwords

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

train['Review Text'] = train['Review Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
train.head(3)
xtrain, xvalid, ytrain, yvalid = train_test_split(train['Review Text'].values, y, 

                                                  stratify=y, 

                                                  random_state=42, 

                                                  test_size=0.1, shuffle=True)
print (xtrain.shape)

print (xvalid.shape)
xtrain
# Always start with these features. They work (almost) everytime!

tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')



# Fitting TF-IDF to both training and test sets (semi-supervised learning)

tfv.fit(list(xtrain) + list(xvalid))

xtrain_tfv =  tfv.transform(xtrain) 

xvalid_tfv = tfv.transform(xvalid)
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
# Fitting a simple Logistic Regression on TFIDF

clf = LogisticRegression(C=1.0)

clf.fit(xtrain_tfv, ytrain)

predictions = clf.predict_proba(xvalid_tfv)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
# Fitting a simple xgboost on tf-idf

clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

clf.fit(xtrain_tfv.tocsc(), ytrain)

predictions = clf.predict_proba(xvalid_tfv.tocsc())



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
import pickle

pickle_off = open("/kaggle/input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl", "rb")

emp = pickle.load(pickle_off)
type(emp)
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = emp
def sent2vec(s):

    words = str(s).lower()

    words = word_tokenize(words)

    words = [w for w in words if not w in stop_words]

    words = [w for w in words if w.isalpha()]

    M = []

    for w in words:

        try:

            M.append(embeddings_index[w])

        except:

            continue

    M = np.array(M)

    v = M.sum(axis=0)

    if type(v) != np.ndarray:

        return np.zeros(300)

    return v / np.sqrt((v ** 2).sum())
# create sentence vectors using the above function for training and validation set

xtrain_glove = [sent2vec(x) for x in tqdm(xtrain)]

xvalid_glove = [sent2vec(x) for x in tqdm(xvalid)]
xtrain_glove = np.array(xtrain_glove)

xvalid_glove = np.array(xvalid_glove)
# Fitting a simple xgboost on glove features

clf = xgb.XGBClassifier(nthread=10, silent=False)

clf.fit(xtrain_glove, ytrain)

predictions = clf.predict_proba(xvalid_glove)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))