# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
tqdm.pandas()
%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import gensim.models.keyedvectors as word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D, Dropout, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import os
from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from scipy.sparse import coo_matrix
from nltk.stem.porter import *

from nltk.stem import WordNetLemmatizer, SnowballStemmer
stopWords = set(stopwords.words('english'))

df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv') 
import re
stemmer = SnowballStemmer(language='english')
def clean_text(x):
    x = str(x).lower()
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x
def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}',"##",x)
    return x

def lemma_stema(sentence):
    new_sent = ""
    for word in sentence.split(" "):
        new_sent = new_sent + " " + stemmer.stem(WordNetLemmatizer().lemmatize(word, pos='v'))
    new_sent = new_sent.strip(" ")
    return new_sent
def remove_stopwords(sentence):
    new_sent = ""
    for word in sentence.split(" "):
        if word not in stopWords:
            new_sent = new_sent + " " + word
    new_sent = new_sent.strip(" ")
    return new_sent
df["question_text"] = df["question_text"].progress_apply(lambda x: clean_numbers(x))
test["question_text"] = test["question_text"].progress_apply(lambda x: clean_numbers(x))

df["question_text"] = df["question_text"].progress_apply(lambda x: clean_text(x))
test["question_text"] = test["question_text"].progress_apply(lambda x: clean_text(x))

df["question_text"] = df["question_text"].progress_apply(lambda x: lemma_stema(x))
test["question_text"] = test["question_text"].progress_apply(lambda x: lemma_stema(x))

df["question_text"] = df["question_text"].progress_apply(lambda x: remove_stopwords(x))
test["question_text"] = test["question_text"].progress_apply(lambda x: remove_stopwords(x))
from sklearn.feature_extraction.text import TfidfVectorizer
tfvecor = TfidfVectorizer(strip_accents='unicode', ngram_range=(1,2))
tfvecor.fit(df['question_text'].append(test['question_text']))
dfx = tfvecor.transform(df['question_text'])
testx = tfvecor.transform(test['question_text'])
# len(tfvecor.get_feature_names())
dfx.shape, testx.shape
# from sklearn.model_selection import train_test_split
# # from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


# X_train, X_test, y_train, y_test = train_test_split(dfx,
#                                                     df['target'],
#                                                     test_size=0.2)
def downsample(df):
    print(df.shape)
    np.random.seed=42
    dfzero = df[df['target']==0]
    dfone = df[df['target']==1]
    zeroind = np.random.choice(dfzero.index,100000, replace=False)
    dfzerodown = dfzero.loc[zeroind, : ]
    dffinal = pd.concat([dfzerodown, dfone], axis=0)
    return dffinal
def get_imp_feature(df):
    from sklearn.linear_model import LogisticRegression
    dfx = tfvecor.transform(df['question_text'])
    clf=LogisticRegression(C=0.76,multi_class='multinomial',penalty='l1', solver='saga',n_jobs=-1)
    clf.fit(dfx, df['target'])
    feature_imp = clf.coef_.nonzero()[1]
#     dfimp = dfx[: ,feature_imp]
#     dfimp.todense()
    # clf.fit(X_train, y_train)
    # #clf = MultinomialNB().fit(X_train, y_train)
    # predicted = clf.predict(X_test)
    return feature_imp
df_for_ft= downsample(df)
# df_for_ft_x = tfvecor.transform(df_for_ft['question_text'])

feature_imp = get_imp_feature(df_for_ft)
# get the important feature from the df

len(feature_imp)
dfimp = dfx[: ,feature_imp]
testimp = testx[:, feature_imp]
# dfimp.todense()
from lightgbm import LGBMClassifier
model = LGBMClassifier(max_depth=20, num_leaves=50, learning_rate=0.1, n_jobs=-1, n_estimators=500, feature_fraction= 0.05,
                       bagging_fraction=0.8)
model.fit(dfimp, df['target'])
#predcv = model.predict(X_test)
predicted_test = model.predict(testimp)

test['prediction'] = predicted_test
submission = test.drop(columns=['question_text'])
submission.head()
submission.to_csv('submission.csv', index=False)