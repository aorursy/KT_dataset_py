# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
raw_train = pd.read_csv('../input/train.csv')
raw_test = pd.read_csv('../input/test.csv')
# print(raw_train.info())
# print(raw_test.info())

pNum = (raw_train.target==1).sum()
nNum = (raw_train.target==0).sum()
print('pNum:\t', pNum)
print('nNum:\t', nNum)
print('ratio:\t', pNum/(pNum+nNum))

train_text = raw_train['question_text']
test_text = raw_test['question_text']
all_text = pd.concat([train_text, test_text])

count_vectorizer = CountVectorizer(
#     sublinear_tf=True,
    strip_accents='unicode',
#     analyzer='word',
#     token_pattern=r'\w{1,}',
#     stop_words='english',
#     ngram_range=(1, 1),
    max_features=20000)
# )
tfidf_transformer = TfidfTransformer(
#     sublinear_tf=True,
)

count_vectorizer.fit(all_text)
print('CountVectorizer has been fitted.')

tfidf_transformer.fit(count_vectorizer.transform(all_text))
print('TfidfTransformer has been fitted.')

train_text_counts = count_vectorizer.transform(train_text)
test_text_counts = count_vectorizer.transform(test_text)

# train_text_tfidf = tfidf_transformer.transform(train_text_counts)
# test_text_tfidf = tfidf_transformer.transform(test_text_counts)
countsP = train_text_counts[raw_train.loc[raw_train.target==1].index.tolist()].sum(axis=0)
countsN = train_text_counts[raw_train.loc[raw_train.target==0].index.tolist()].sum(axis=0)
importances = (countsP/(countsN+1)).tolist()[0]
importtancesDf = pd.DataFrame({'fn':count_vectorizer.get_feature_names(), 'importances':importances})

selectedFeaturesNames = importtancesDf.sort_values(by='importances', ascending=False).head(20000).fn
selectedFeaturesIndex = selectedFeaturesNames.index.tolist()
train_text_tfidf = tfidf_transformer.transform(train_text_counts)
test_text_tfidf = tfidf_transformer.transform(test_text_counts)

train_text_tfidf = train_text_tfidf[:, selectedFeaturesIndex]
test_text_tfidf = test_text_tfidf[:, selectedFeaturesIndex]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
outTest = np.zeros((test_text_tfidf.shape[0]))
validPre = np.zeros((train_text_tfidf.shape[0]))
for train_index, valid_index in kfold.split(train_text_tfidf, raw_train.target):
    X_train, X_valid = train_text_tfidf[train_index], train_text_tfidf[valid_index]
    y_train, y_valid = raw_train.target[train_index], raw_train.target[valid_index]
    
#     model = LogisticRegression(solver='liblinear', C=1)
    model = lgb.LGBMClassifier(random_state=2018,
                               max_depth=-1,
                               num_leaves=63,
                               n_estimators=300,
                               min_child_samples=30,
                               learning_rate=0.2)
    
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='binary_error',
              early_stopping_rounds=20)
    
    pre = model.predict_proba(X_valid)[:, 1]
    validPre[valid_index] = pre
    
#     score = f1_score(y_valid, pre>0.5)
#     print('score:\t', score)
    
    outTest += model.predict_proba(test_text_tfidf)[:, 1]/5
best_score = 0
best_threshold = 0
for i in tqdm(range(100)):
    Threshold = float(i)/100
    score = f1_score(raw_train.target, validPre>Threshold)
    if score > best_score:
        best_score = score
        best_threshold = Threshold

print('threshold:\t', best_threshold, ', score:\t', best_score)
outDf = pd.DataFrame({'qid': raw_test.qid, 'prediction': outTest>best_threshold})
outDf.to_csv('submission.csv', index=False)