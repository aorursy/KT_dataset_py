# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output
sms = pd.read_csv("../input/spam.csv", encoding='latin-1')

sms.head()
sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)

  
sms = sms.rename(columns = {'v1':'label','v2':'message'})
sms.groupby('label').describe()
sms['length'] = sms['message'].apply(len)

sms.head()
import matplotlib as mpl

import matplotlib.pyplot as plt
mpl.rcParams['patch.force_edgecolor'] = True

plt.style.use('seaborn-bright')

sms.hist(column='length', by='label', bins=50,figsize=(11,5))
text_feat = sms['message'].copy()
import string

from nltk.corpus import stopwords
def text_process(text):

    

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    

    return " ".join(text)
import string

from nltk.corpus import stopwords
text_feat = text_feat.apply(text_process)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer("english")
from sklearn.feature_extraction import DictVectorizer

vectorizer = DictVectorizer()
import requests

%matplotlib inline
features = vectorizer.fit_transform(text_feat)