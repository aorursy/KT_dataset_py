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

import re

import numpy as np

from nltk.tokenize import word_tokenize

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection, naive_bayes

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train = train.drop(['location'], axis = 1)

test = test.drop(['location'], axis = 1)
# Removing null values in train

train = train.dropna()
train['target'].value_counts()
# random data balance

Input_0 = train.loc[train['target'] == 0].sample(3229)

Input_0 = Input_0.reset_index(drop = True)

Input_1 = train.loc[train['target'] == 1]

Input_1 = Input_1.reset_index(drop = True)

Input_balanced = Input_1.append(Input_0)
Input_balanced['target'].value_counts()
train = Input_balanced
# Club train test data

ntrain = train.shape[0]

ntest = test.shape[0]

target = train.target.values

train.drop(['target'], axis=1, inplace=True)

df_combined = pd.concat((train, test)).reset_index(drop=True)

print("df_combined size is : {}".format(df_combined.shape))
# Replace null values in Keyword column

df_combined = df_combined.fillna("Not Known")
df_combined.isnull().sum()   
def clean_text(text):

    text = text.lower()

    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)   

    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)   

    text = re.sub(r'www.[^ ]+', '', text)  

    text = re.sub(r'[a-zA-Z0-9]*www[a-zA-Z0-9]*com[a-zA-Z0-9]*', '', text)  

    text = re.sub(r'[^a-zA-Z]', ' ', text)   

    text = [token for token in text.split() if len(token) > 2]

    text = ' '.join(text)

    return text
df_combined['Keyword_Processed'] = df_combined['keyword'].apply(clean_text)

df_combined['Text_Processed'] = df_combined['text'].apply(clean_text)
df_combined['Key_Text'] = df_combined['Keyword_Processed'] +' ' + df_combined['Text_Processed']
# Splitting Train and Test

train_df = df_combined[:ntrain]

test_df = df_combined[ntrain:]
# Train test split

from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(train_df['Text_Processed'], target, test_size=0.25, stratify=target, 

                                                random_state=1)
# Tf-IDF vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1,2), max_df=1.0, 

                             min_df=3, max_features=None, binary=False, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
X_train_tfidf = tfidf_vect.fit_transform(X_train)

X_cv_tfidf = tfidf_vect.transform(X_cv)
from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()

naive_bayes.fit(X_train_tfidf, y_train)

y_pred_class = naive_bayes.predict(X_cv_tfidf)

train_pred_class = naive_bayes.predict(X_train_tfidf)

print('f1_score       :', f1_score(y_cv, y_pred_class, average='macro'))

print('Train accuracy score :', accuracy_score(y_train, train_pred_class))

print('Test accuracy score :', accuracy_score(y_cv, y_pred_class))

print("=== Confusion Matrix ===")

print(confusion_matrix(y_cv, y_pred_class))

print(classification_report(y_cv, y_pred_class))
from sklearn.metrics import f1_score, accuracy_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='log', max_iter=200, random_state=0, class_weight='balanced')

ovr = OneVsRestClassifier(sgd)

ovr.fit(X_train_tfidf, y_train)

y_pred_class = ovr.predict(X_cv_tfidf)

print('f1_score       :', f1_score(y_cv, y_pred_class, average='macro'))

print('accuracy score :', accuracy_score(y_cv, y_pred_class))
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_jobs = -1, learning_rate = 0.1, max_depth = 32, objective = 'binary:logistic',

                        min_child_weight = 1, subsample = 1, colsample_bytree = 1, n_estimators = 150)

xgb_model.fit(X_train_tfidf, y_train)

y_pred_class = xgb_model.predict(X_cv_tfidf)

train_pred_class = xgb_model.predict(X_train_tfidf)

print('f1_score       :', f1_score(y_cv, y_pred_class, average='macro'))

print('Train accuracy score :', accuracy_score(y_train, train_pred_class))

print('Test accuracy score :', accuracy_score(y_cv, y_pred_class))

print("=== Confusion Matrix ===")

print(confusion_matrix(y_cv, y_pred_class))

print(classification_report(y_cv, y_pred_class))
# Prediction on test set

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, 

                             min_df=3, max_features=None, binary=False, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
full_text = list(train_df['Key_Text'].values) + list(test_df['Key_Text'].values)

tfidf_vect.fit(full_text)
X_train_tfidf = tfidf_vect.transform(train_df['Key_Text'])

X_test_tfidf = tfidf_vect.transform(test_df['Key_Text'])
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='log', max_iter=200, random_state=0, class_weight='balanced')

ovr = OneVsRestClassifier(sgd)

ovr.fit(X_train_tfidf, target)

Preds_OVR = ovr.predict(X_test_tfidf)

Preds_OVR
naive_bayes = MultinomialNB()

naive_bayes.fit(X_train_tfidf, target)

Preds_NB = naive_bayes.predict(X_test_tfidf)

Preds_NB
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_jobs = -1, learning_rate = 0.1, max_depth = 32, objective = 'binary:logistic',

                         min_child_weight = 0.9, subsample = 0.9, colsample_bytree = 0.9, n_estimators = 150)

xgb_model.fit(X_train_tfidf, target)

Preds_XGB = xgb_model.predict(X_test_tfidf)

Preds_XGB
# Saving XGBoost predictions

Submission = pd.DataFrame()

Submission['id'] = test_df['id']

Submission['target'] = Preds_XGB
Submission.to_csv("Real_Or_Not.csv", index = False)