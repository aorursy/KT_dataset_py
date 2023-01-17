

import numpy as np

import pandas as pd 

import warnings

import re

import matplotlib.pyplot as plt

import random



from Levenshtein import *

from nltk.corpus import *

from nltk import ChartParser

from subprocess import check_output



from sklearn.preprocessing import MinMaxScaler





warnings.filterwarnings('ignore')

print(check_output(["ls", "../input"]).decode("utf8"))
test = pd.read_csv('../input/test.csv', encoding='ISO-8859-1').fillna("")

train = pd.read_csv('../input/train.csv', encoding ='ISO-8859-1').fillna("")


unique_ids1 = train['id'].unique()

rand_ids1 = [unique_ids1[i] for i in sorted(random.sample(range(len(unique_ids1)), 1000)) ]

train = train[train.id.isin(rand_ids1)]



train.shape
unique_ids2 = test['test_id'].unique()

rand_ids2 = [unique_ids2[i] for i in sorted(random.sample(range(len(unique_ids2)), 1000)) ]

test = test[test.test_id.isin(rand_ids2)]



test.shape


stopwords = stopwords.words('english')

print(stopwords)
test.info()



train.info()
test.head()
train.head()
train.groupby("is_duplicate")['id'].count().plot.bar()
train = train.dropna()              

train['question1'] = train['question1'].apply(lambda x: x.rstrip('?'))

train['question2'] = train['question2'].apply(lambda x: x.rstrip('?'))

train['question1'] = train['question1'].str.lower().str.split(' ')

train['question2'] = train['question2'].str.lower().str.split(' ')

train['question1'] = train['question1'].apply(lambda x: [item for item in x if item not in stopwords])

train['question2'] = train['question2'].apply(lambda x: [item for item in x if item not in stopwords])
#test = test.dropna()     

test['question1'] = test['question1'].apply(lambda x: x.rstrip('?'))

test['question2'] = test['question2'].apply(lambda x: x.rstrip('?'))

test['question1'] = test['question1'].str.lower().str.split(' ')

test['question2'] = test['question2'].str.lower().str.split(' ')

test['question1'] = test['question1'].apply(lambda x: [item for item in x if item not in stopwords])

test['question2'] = test['question2'].apply(lambda x: [item for item in x if item not in stopwords])
train['Common'] = train.apply(lambda row: len(list(set(row['question1']).intersection(row['question2']))), axis=1)

test['Common'] = test.apply(lambda row: len(list(set(row['question1']).intersection(row['question2']))), axis=1)



train['Average'] = train.apply(lambda row: 0.5*(len(row['question1'])+len(row['question2'])), axis=1)

test['Average'] = test.apply(lambda row: 0.5*(len(row['question1'])+len(row['question2'])), axis=1)



train['Percentage'] = train.apply(lambda row: row['Common'] * 100.0 / (row['Average'] + 1), axis=1)

test['Percentage'] = test.apply(lambda row: 1 if row['Average'] == 0 else row['Common']/(row['Average']), axis=1)



train['seqratio'] = train.apply(lambda row: seqratio(row['question1'], row['question2']) * 100.0, axis=1)

test['seqratio'] = test.apply(lambda row: seqratio(row['question1'], row['question2']) * 100.0, axis=1)



train['avg2'] = train.apply(lambda row: (row['seqratio'] + row['Percentage']) / 200.0, axis=1)

test['avg2'] = test.apply(lambda row: (row['seqratio'] + row['Percentage']) / 200.0, axis=1)
train.head()
train.corr()['is_duplicate']




# Check these for later

#dfTrain = pd.DataFrame({'test_id' : range(0,2345796)})

#dfTest = pd.DataFrame({'test_id' : range(0,2345796)})

#dfTrain[['Common', 'Average', 'Percentage', 'seqratio', 'avg2']] = scaler.fit_transform(train[['Common', 'Average', 'Percentage', 'seqratio', 'avg2']])

#dfTest[['Common', 'Average', 'Percentage', 'seqratio', 'avg2']] = scaler.fit_transform(test[['Common', 'Average', 'Percentage', 'seqratio', 'avg2']])



#df = pd.DataFrame({'test_id' : range(0,2345796)})

#df.fillna(0, inplace = True)

#df['is_duplicate'] = pd.Series(test['avg2'])



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_curve, auc, roc_curve

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler().fit(train[['Common', 'Average', 'Percentage', 'seqratio', 'avg2']])



X = scaler.transform(train[['Common', 'Average', 'Percentage', 'seqratio', 'avg2']])

y = train['is_duplicate']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)



X_train.shape, X_test.shape, y_train.shape, y_test.shape



clf = LogisticRegression()

grid = {

    'C': [1e-6, 1e-3, 1e0],

    'penalty': ['l1', 'l2']

}

cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=-1, verbose=1)

cv.fit(X_train, y_train)



retrained = cv.best_estimator_.fit(X, y)



scaler = MinMaxScaler().fit(test[['Common', 'Average', 'Percentage', 'seqratio', 'avg2']])



X_submission = scaler.transform(test[['Common', 'Average', 'Percentage', 'seqratio', 'avg2']])

y_submission = retrained.predict_proba(X_submission)[:,1]



dftest = pd.read_csv("../input/test.csv").fillna("")

submission = pd.DataFrame({'test_id': dftest['test_id'], 'is_duplicate': y_submission})

submission.head()



print(submission.shape)

submission.to_csv('submit.csv', index=False)
submission.head(10)