# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Data Loading and Preview

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print(train.shape)
print(train.columns)
print(train[['text','target']].sample(20))
#Investigating Data Properties
print(len(train['keyword'].unique()))
print(len(train['location'].unique()))
print(train['target'].value_counts())
print(train['target'].value_counts(normalize = True))

print(train[train.target == 1]['text'].sample(5).values)
print(train[train.target == 0]['text'].sample(5).values)
pd.crosstab(train['text'].apply(lambda x: 1 if 'dead' in x.lower() or 'kill' in x.lower() else 0), train['target'])
pd.crosstab(train['text'].apply(lambda x: 1 if 'http' in x.lower() else 0), train['target'], normalize = True)

#counting words in each tweet
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train['text'])
print(train_vectors.shape)
print(count_vectorizer.get_feature_names()[:20])

test_vectors = count_vectorizer.transform(test['text'])
train_corpus_non_stand = pd.DataFrame(train_vectors.todense())
train_corpus_non_stand.columns = count_vectorizer.get_feature_names()
train_corpus_non_stand.head()

test_corpus_non_stand = pd.DataFrame(test_vectors.todense())
test_corpus_non_stand.columns = count_vectorizer.get_feature_names()
test_corpus_non_stand
train_corpus_non_stand.sum().sort_values(ascending = False).head(20)

#too many stopword in our vector list. removing them using nltk
import nltk
from nltk.corpus import stopwords
print(len(stopwords.words('english')))
def get_normalized_values(train_idv,test_idv):
    mean,sd,train_idv,test_idv=get_mean_sd(train_idv,test_idv)
    idvs=list(train_idv.columns)
    train_x=pd.DataFrame()
    test_x=pd.DataFrame()
    for idv in idvs:
        train_x[idv]=train_idv[idv].apply(lambda m: (m-mean[idv])/sd[idv])
        test_x[idv]=test_idv[idv].apply(lambda n: (n-mean[idv])/sd[idv])
    return train_x,test_x

def get_mean_sd(train_idv,test_idv):
    mean=np.mean(train_idv)
    sd=np.std(train_idv)
    idvs=list(mean.index)
    for idv in idvs:
        if sd[idv]==0:
            train_idv.drop([idv],axis=1,inplace=True)
            test_idv.drop([idv],axis=1,inplace=True)
        else:
            train_idv[idv].fillna(mean[idv],inplace=True)
            test_idv[idv].fillna(mean[idv],inplace=True)
    return mean,sd,train_idv,test_idv

#train_stand, test_stand = get_normalized_values(train_corpus_non_stand, test_corpus_non_stand)
train_stand = (train_corpus_non_stand - train_corpus_non_stand.mean())/train_corpus_non_stand.std()
test_stand = (test_corpus_non_stand - test_corpus_non_stand.mean())/train_corpus_non_stand.std()

LR = linear_model.LogisticRegression
c_values = [0.001, 0.01, 0.1, 1]
c_values = [0.1]

for c_value in c_values:
    model=LR(penalty='l2',C=c_value, solver = 'liblinear')
    model.fit(train_stand,train['target'].values)
    scores = model_selection.cross_val_score(model, train_vectors, train["target"], cv=3, scoring="f1")
    print(scores)
    print("for c_value = ", c_value, "scores = ", scores)
RC = linear_model.RidgeClassifier

model=RC()
scores = model_selection.cross_val_score(model, train_vectors, train["target"], cv=3, scoring="f1")
print(scores)
print("for c_value = ", c_value, "scores = ", scores)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = model.predict(test_vectors)
sample_submission.to_csv("RF_submission.csv", index=False)