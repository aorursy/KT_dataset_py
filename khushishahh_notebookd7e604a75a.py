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
sms = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
import matplotlib.pyplot as plt

import nltk

import re

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

import string
sms.head()
sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

sms = sms.rename(columns = {'v1':'label','v2':'message'})



sms['length'] = sms['message'].apply(len)

sms.head()

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import string

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords



sms = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')

sms.head()



sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

sms = sms.rename(columns = {'v1':'label','v2':'message'})



sms['length'] = sms['message'].apply(len)

sms.head()



text_feat = sms['message'].copy()

def text_process(text):

    

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    

    return " ".join(text)

text_feat = text_feat.apply(text_process)

vectorizer = TfidfVectorizer("english")

features = vectorizer.fit_transform(text_feat)



features_train, features_test, labels_train, labels_test = train_test_split(features, sms['label'],

                                                                            test_size=0.3, random_state=111)

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score



mnb = MultinomialNB(alpha=0.2)

mnb.fit(features_train,labels_train)

pred=mnb.predict(features)
