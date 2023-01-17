import numpy as np 

import pandas as pd

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, classification_report, f1_score

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df  = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
vect = TfidfVectorizer()

sklearn_tokenizer = vect.build_tokenizer()

stop_words = set(stopwords.words("english"))

y = df.target.to_numpy()

vect=TfidfVectorizer(tokenizer = sklearn_tokenizer,stop_words='english',ngram_range=(1, 1), norm='l2')

clf=SGDClassifier(alpha=0.0001,epsilon=0.1, eta0=0.0,

                               l1_ratio=0.1, learning_rate='optimal',

                               loss='modified_huber', penalty='l2',class_weight =  'balanced')

pp = Pipeline([('vect',vect),('clf',clf )])
pp.fit(df.question_text.to_numpy(),y)
test_df =  pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
test_df['prediction'] = pp.predict(test_df.question_text.to_numpy())

test_df[['qid','prediction']].to_csv("submission.csv", index=False)