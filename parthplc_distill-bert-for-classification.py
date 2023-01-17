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

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
df_sentences = pd.read_csv('../input/sentiment-labelled-sentences-data-set/amazon_cells_labelled.txt', sep="\t", header=None)
df_sentences.head()
import string

import re

def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
train.head()

train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))



# Let's take a look at the updated text

train['text'].head()
print(train.shape,

test.shape)
train.head()

train.drop(['id','keyword','location'],axis = 1,inplace = True)
train.head()
# Import libraries
import numpy as np

import pandas as pd

import torch

import transformers as ppb # pytorch transformers

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
train.head(20)
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

model = model_class.from_pretrained(pretrained_weights)
tokenized = train["text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized.shape
tokenized
max_len = 0

for i in tokenized.values:

    if len(i) > max_len:

        max_len = len(i)



padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
np.array(padded).shape

attention_mask = np.where(padded != 0, 1, 0)

attention_mask.shape
#Model #1: And Now, Deep Learning!¶

input_ids = torch.tensor(padded)  

attention_mask = torch.tensor(attention_mask)



with torch.no_grad():

    last_hidden_states = model(input_ids, attention_mask=attention_mask)
features = last_hidden_states[0][:,0,:].numpy()

labels = train['target']
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
#Grid Search for Parameters
from sklearn.model_selection import GridSearchCV



parameters = {'C': np.linspace(0.01, 100, 20)}

grid_search = GridSearchCV(LogisticRegression(), parameters)

grid_search.fit(train_features, train_labels)



print('best parameters: ', grid_search.best_params_)

print('best scrores: ', grid_search.best_score_)
lr_clf = LogisticRegression()

lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels)

from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
# Fitting a simple Naive Bayes on Counts

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)

clf.fit(train_features, train_labels)



clf.score(test_features, test_labels)
