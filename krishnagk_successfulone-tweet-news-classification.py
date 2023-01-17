#Reference: https://www.kaggle.com/yosukehasimoto/real-or-not/comments

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
import os

import math

import datetime



from tqdm import tqdm



import pandas as pd

import numpy as np



import tensorflow as tf

from tensorflow import keras



#import bert

#from bert import BertModelLayer

#from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

#from bert.tokenization.bert_tokenization import FullTokenizer



import seaborn as sns

from pylab import rcParams

import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from matplotlib import rc



from sklearn.metrics import confusion_matrix, classification_report



import string

import re



from keras.models import model_from_json



train = pd.read_csv('../input/nlp-getting-started/train.csv')

submission = pd.read_csv('../input/nlp-getting-started/test.csv')

test = submission

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

train.head(5)



print(test)


#from sklearn.model_selection import train_test_split



#train, test = train_test_split(train, test_size=0.4)



#X_train, X_test, y_train, y_test = train_test_split(train.text, train.target, test_size=0.25, random_state=1000)


#from sklearn.model_selection import train_test_split

#from sklearn.preprocessing import LabelEncoder

#from keras.models import Model

#from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

#from keras.optimizers import RMSprop

#from keras.preprocessing.text import Tokenizer

#from keras.preprocessing import sequence

#from keras.utils import to_categorical

#from keras.callbacks import EarlyStopping

#%matplotlib inline

#import re





#X = train.text

#Y = train.target

#le = LabelEncoder()

#Y = le.fit_transform(Y)

#Y = Y.reshape(-1,1)

#print(X)

#print(Y)

#import pandas as pd

#import numpy as np

#from sklearn.feature_extraction.text import TfidfVectorizer

#from sklearn.linear_model.logistic import LogisticRegression

#from sklearn.model_selection import train_test_split, cross_val_score

#train = pd.read_csv('../input/nlp-getting-started/train.csv')

#submission = pd.read_csv('../input/nlp-getting-started/test.csv')

#test = submission

#sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

#train.head(5)



#vectorizer = TfidfVectorizer(stop_words='english')

#tfidf_train_x = vectorizer.fit_transform(train['text'])

#classifier = LogisticRegression()

#classifier.fit(tfidf_train_x, train['target'])
#tfidf_test_x = vectorizer.transform(test['text'])

#print (tfidf_test_x.shape)

#scores = cross_val_score(classifier, tfidf_test_x, test['id'])

#acc = scores.mean()

#print ("Accuracy: %0.2f percent" % (acc *100))
sample_submission.shape
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer



vectorizer = CountVectorizer(analyzer='word', binary=True)

vectorizer.fit(train['text'])
X = vectorizer.transform(train['text']).todense()

y = train['target'].values

X.shape, y.shape
from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split



from sklearn.metrics import f1_score



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
model = LogisticRegression()

model.fit(X_train, y_train)



#model = LogisticRegression()

y_pred = model.predict(X_test)



f1score = f1_score(y_test, y_pred)

print(f1score)
tweets_test = test['text']

test_X = vectorizer.transform(tweets_test).todense()

test_X.shape
lr_pred = model.predict(test_X)
sample_submission['target'] = lr_pred

sample_submission.to_csv("submission.csv", index=False)

sample_submission.head()

sample_submission.shape