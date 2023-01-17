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

from tqdm import tqdm

from collections import Counter

import re

from nltk.corpus import stopwords

import string

from keras.utils import to_categorical

import nltk

nltk.download('wordnet')

nltk.download('stopwords')

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

tqdm.pandas()
df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', engine='python',usecols=range(2),names=['class','text'], skiprows=[0])

df.head()
df.info()
df['class'] = df['class'].rank(method='dense', ascending=False).astype(int)
def clean_data(text):

   lower = text.lower()

   splitted = lower.split()

   re_punc = re.compile('[%s]' % re.escape(string.punctuation))

   tokens = [re_punc.sub('',w) for w in splitted]

   tokens = [word for word in tokens if word.isalpha()]

   stop_words = set(stopwords.words('english'))

   tokens = [w for w in tokens if not w in stop_words]

   lemmeted = [WordNetLemmatizer().lemmatize(w) for w in tokens]

   tokens = [word for word in lemmeted if len(word) > 2]

   return tokens



x_train, x_val, y_train, y_val = train_test_split(df['text'], df['class'])

vocab = Counter()

for index, row in x_train.iteritems():

  vocab.update(clean_data(row))
vocab.most_common(20)
vocab_size = len(vocab)

print("Vocabulary Size is: ", vocab_size)
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer(vocabulary=vocab.keys())

x_train = v.fit_transform(x_train)

x_val = v.fit_transform(x_val)
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import classification_report



model = BernoulliNB()

model.fit(x_train, y_train)

pred = model.predict(x_val)

print(classification_report(y_val, pred))