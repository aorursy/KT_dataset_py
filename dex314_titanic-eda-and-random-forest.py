# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', index_col=0)
train.head(10)
test = pd.read_csv('../input/test.csv', index_col=0)
test.head(10)
## this is just the sample submission
sample_sub = pd.read_csv('../input/gender_submission.csv')
# gs.head(10)
train.info()
train.Embarked.value_counts()
train.Sex.value_counts()
# train.Cabin.value_counts()
print(train.drop(['Survived'],axis=1).describe())
print('\n========================================')
print(test.describe())
print("The test set is approximately {:.3f} % if theyre counted as a whole together".format( 100*len(test) / (len(train)+len(test)) ))
train.isnull().sum()/len(train)
test.isnull().sum()/len(test)
target = train.Survived
target.value_counts()
print("The survival ratio is about {:.3f}%".format(target.value_counts()[1]/len(target) *100))
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
fig, ax=plt.subplots(1,1,figsize=(12,10))
sns.swarmplot(x="Age", y="Sex", hue="Survived", data=train, ax=ax)
plt.title("Age v Sex on Survival");
train_nc = train.copy()
train_nc['Sex'] = pd.factorize(train.Sex)[0]
dummies = pd.get_dummies(train.Embarked)
print(dummies.head(3))
train_nc = train_nc.join(dummies)
print(train_nc.head(3))
train_nc.drop('Embarked',axis=1,inplace=True)
fig, ax=plt.subplots(1,1,figsize=(12,12))
sns.heatmap(train_nc.corr(),annot=True,cmap='coolwarm');
fig, ax=plt.subplots(1,1,figsize=(12,10))
sns.swarmplot(x="Age", y="Sex", hue="Survived", data=train, ax=ax)
plt.title("Age v Sex on Survival");

fig, ax=plt.subplots(1,1,figsize=(12,10))
sns.swarmplot(x="Fare", y="Sex", hue="Survived", data=train, ax=ax)
plt.title("Fare v Sex on Survived");
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import re
import string
from tqdm import tqdm
TOKENIZER = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return TOKENIZER.sub(r' \1 ', s).split()
tfidf = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize, 
                        max_features = 100,
                        strip_accents='unicode', use_idf=True,
                        smooth_idf=True, sublinear_tf=True)
train['Cabin'] = train.Cabin.fillna('None')
test['Cabin'] = test.Cabin.fillna('None')
train_name = tfidf.fit_transform(train['Name'])
train_ticket = tfidf.fit_transform(train['Ticket'])
train_cabin = tfidf.fit_transform(train['Cabin'])

train_tf = np.concatenate((train_name.todense(), train_ticket.todense(), train_cabin.todense()), axis=1)

train_full = train_nc.join(pd.DataFrame(train_tf, index=train.index))
train_full.shape
test_nc = test.copy()
test_nc['Sex'] = pd.factorize(test.Sex)[0]
dummies = pd.get_dummies(test.Embarked)
print(dummies.head(3))
test_nc = test_nc.join(dummies)
print(test_nc.head(3))
test_nc.drop('Embarked',axis=1,inplace=True)
test_name = tfidf.fit_transform(test['Name'])
test_ticket = tfidf.fit_transform(test['Ticket'])
test_cabin = tfidf.fit_transform(test['Cabin'])

test_tf = np.concatenate((test_name.todense(), test_ticket.todense(), test_cabin.todense()), axis=1)

test_full = test_nc.join(pd.DataFrame(test_tf, index=test.index))
test_full.shape
print("DROPPING THE CAT VARS...")
train_full.drop(['Name','Ticket','Cabin','Survived'],axis=1, inplace=True)
test_full.drop(['Name','Ticket','Cabin'],axis=1, inplace=True)
train_full = train_full.fillna(0)
test_full = test_full.fillna(0)

# ss = StandardScaler(with_mean=True, with_std=True)
## does better without the scaler

train_fss = train_full #ss.fit_transform(train_full)
test_fss = test_full #ss.fit_transform(test_full)
rf = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=2, random_state=42)
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result
kfolds = 3
preds = 0
ths = []
print(" ====================================== ")
for k in range(kfolds):
    xt, xv, yt, yv = train_test_split(train_full, target, test_size=0.15, random_state=(k+1)*42)
    print(xt.shape, xv.shape, yt.shape, yv.shape)
    rf.fit(xt, yt)
    val_preds = rf.predict_proba(xv)
    print(classification_report(yv, np.round(val_preds[:,1],0).astype(int)))
    th_sr = threshold_search(yv, np.round(val_preds[:,1],0).astype(int))
    ths.append(th_sr)
    print("Threshold Search F1 Result :")
    print(th_sr)
    print(" ========================================== ")
    preds += rf.predict(test_fss)

preds = preds/kfolds
print(preds.shape)
sample_sub['Survived'] = np.round(preds,0).astype(int)
sample_sub.to_csv('submission.csv', index=False)
print(sample_sub.Survived.value_counts())
print(sample_sub.head(10))
print(target.value_counts())

