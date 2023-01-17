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
#Data analysis

import pandas as pd

import numpy as np

#Data visualisation

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)

sns.set(font_scale=1)

%matplotlib inline

%config InlineBackend.figure_format = 'svg'

#Modeling

from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.ensemble import RandomForestClassifier

# from sklearn_crfsuite import CRF, scorers, metrics

# import sklearn_crfsuite

# from sklearn_crfsuite import scorers

# from sklearn_crfsuite import metrics

# from sklearn_crfsuite.metrics import flat_classification_report

from sklearn.metrics import classification_report, make_scorer

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

import scipy.stats

import eli5
df = pd.read_csv('/kaggle/input/entity-annotated-corpus/ner_dataset.csv', encoding="latin1")
#The dataset does not have any header currently. We can use the first row as a header as it has the relevant headings.

#We will make the first row as the heading, remove the first row and re-index the dataset



df.columns = df.iloc[0]



df = df[1:]



df.columns = ['Sentence #','Word','POS','Tag']



df = df.reset_index(drop=True)



df.head()

df = df.rename(columns={"Sentence #": "sentence#"})
df.head()
df.shape
df.info()
#so we are basically having only those rows where sentence column is not null

data = df[df['sentence#'].notnull()]
data.info()
data.head()
# A class to retrieve the sentences from the dataset

class getsentence(object):

    

    def __init__(self, data):

        self.n_sent = 1.0

        self.data = data

        self.empty = False

        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),

                                                           s["POS"].values.tolist(),

                                                           s["Tag"].values.tolist())]

        self.grouped = self.data.groupby("sentence#").apply(agg_func)

        self.sentences = [s for s in self.grouped]
data.head()
getter = getsentence(data)
sentences = getter.sentences

#ths is how a sentence will look like.

print(sentences[1])
#Lets find the number of words in the dataset

words = list(set(data["Word"].values))

n_words = len(words)

print(n_words)
#Lets visualize how the sentences are distributed by their length

plt.style.use("ggplot")

plt.hist([len(s) for s in sentences], bins=50)

plt.show()
#Lets find out the longest sentence length in the dataset

maxlen = max([len(s) for s in sentences])

print ('Maximum sentence length:', maxlen)
#Words tagged as B-org

data.loc[data['Tag'] == 'B-org', 'Word'].head()
#Words tagged as I-org

data.loc[data['Tag'] == 'I-org', 'Word'].head()
#Words tagged as B-per

data.loc[data['Tag'] == 'B-per', 'Word'].head()
#Words tagged as I-per

data.loc[data['Tag'] == 'I-per', 'Word'].head()
#Words tagged as B-geo

data.loc[data['Tag'] == 'B-geo', 'Word'].head()

#Words tagged as I-geo

data.loc[data['Tag'] == 'I-geo', 'Word'].head()
#Words tagged as I-geo

data.loc[data['Tag'] == 'O', 'Word'].head()
#Words distribution across Tags

plt.figure(figsize=(15, 5))

ax = sns.countplot('Tag', data=data)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

plt.tight_layout()

plt.show()
#Words distribution across Tags without O tag

plt.figure(figsize=(15, 5))

ax = sns.countplot('Tag', data=data.loc[data['Tag'] != 'O'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

plt.tight_layout()

plt.show()
#Words distribution across POS

plt.figure(figsize=(15, 5))

ax = sns.countplot('POS', data=data, orient='h')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

plt.tight_layout()

plt.show()
#Simple feature map to feed arrays into the classifier. 

def feature_map(word):

    return np.array([word.istitle(), word.islower(), word.isupper(), len(word),

                     word.isdigit(),  word.isalpha()])
#We divide the dataset into train and test sets

words = [feature_map(w) for w in data["Word"].values.tolist()]

tags = data["Tag"].values.tolist()
#Lets see how the input array looks like

print(words[:5])
#Random Forest classifier

pred = cross_val_predict(RandomForestClassifier(n_estimators=20),X=words, y=tags, cv=5)
#Lets check the performance 

from sklearn.metrics import classification_report

report = classification_report(y_pred=pred, y_true=tags)

print(report)
!pip install sklearn-crfsuite
from itertools import chain



import nltk

import sklearn

import scipy.stats



import sklearn_crfsuite

from sklearn_crfsuite import scorers,CRF

from sklearn_crfsuite.metrics import flat_classification_report

from sklearn_crfsuite import metrics
# Feature set

def word2features(sent, i):

    word = sent[i][0]

    postag = sent[i][1]



    features = {

        'bias': 1.0,

        'word.lower()': word.lower(),

        'word[-3:]': word[-3:],

        'word[-2:]': word[-2:],

        'word.isupper()': word.isupper(),

        'word.istitle()': word.istitle(),

        'word.isdigit()': word.isdigit(),

        'postag': postag,

        'postag[:2]': postag[:2],

    }

    if i > 0:

        word1 = sent[i-1][0]

        postag1 = sent[i-1][1]

        features.update({

            '-1:word.lower()': word1.lower(),

            '-1:word.istitle()': word1.istitle(),

            '-1:word.isupper()': word1.isupper(),

            '-1:postag': postag1,

            '-1:postag[:2]': postag1[:2],

        })

    else:

        features['BOS'] = True



    if i < len(sent)-1:

        word1 = sent[i+1][0]

        postag1 = sent[i+1][1]

        features.update({

            '+1:word.lower()': word1.lower(),

            '+1:word.istitle()': word1.istitle(),

            '+1:word.isupper()': word1.isupper(),

            '+1:postag': postag1,

            '+1:postag[:2]': postag1[:2],

        })

    else:

        features['EOS'] = True



    return features
def sent2features(sent):

    return [word2features(sent, i) for i in range(len(sent))]



def sent2labels(sent):

    return [label for token, postag, label in sent]
#Creating the train and test set

X = [sent2features(s) for s in sentences]

y = [sent2labels(s) for s in sentences]
#Creating the CRF model

crf = CRF(algorithm='lbfgs',

          c1=0.1,

          c2=0.1,

          max_iterations=100,

          all_possible_transitions=False)
#We predcit using the same 5 fold cross validation

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
#Lets evaluate the mode

report = flat_classification_report(y_pred=pred, y_true=y)

print(report)
#Tuning the parameters manually, setting c1 = 10

crf2 = CRF(algorithm='lbfgs',

          c1=10,

          c2=0.1,

          max_iterations=100,

          all_possible_transitions=False)
pred = cross_val_predict(estimator=crf2, X=X, y=y, cv=5)

report = flat_classification_report(y_pred=pred, y_true=y)

print(report)