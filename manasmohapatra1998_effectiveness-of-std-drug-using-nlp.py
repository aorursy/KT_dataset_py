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
#importing all the library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
train = pd.read_csv("/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/train.csv")
test = pd.read_csv("/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/train.csv")
train.head()
#checking for null
train.isna().sum()
# Taking the required columns
train_new=train[['patient_id','effectiveness_rating','number_of_times_prescribed','review_by_patient','base_score']]
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import sentiwordnet as swn, wordnet
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
from nltk import pos_tag, word_tokenize
lemmatizer = WordNetLemmatizer()
def lemmatize_words(review_by_patient):
    final_text = []
    for i in review_by_patient.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text)
train_new.review_by_patient = train_new.review_by_patient.apply(lemmatize_words)
train_new.head()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_pandas import DataFrameMapper, cross_val_score
data2 = train_new.copy()
data2 = data2.fillna('')
mapper = DataFrameMapper([
     ('patient_id',None),
     ('effectiveness_rating',None),
     ('number_of_times_prescribed', None),
     ('review_by_patient', TfidfVectorizer()),
 ])
features = mapper.fit_transform(data2)
features.shape
train_new.dtypes
pred_base_score = train_new['base_score']
from sklearn.model_selection import train_test_split
# Split the data between train and test
x_train, x_test, y_train, y_test = train_test_split(features,pred_base_score,test_size=0.2,train_size=0.8, random_state = 0)

x_train.shape
x_test.shape
y_train
y_train_new=y_train
y_train_new.shape
from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train_new)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
x_train.shape
x_hlf=x_train[0:500, :]
x_hlf.shape
encoded.shape
y_hlf=encoded[0:500]
y_hlf.shape
model.fit(x_hlf,y_hlf)
prediction=model.predict(x_test)
final_prediction=(prediction/100)
final_prediction