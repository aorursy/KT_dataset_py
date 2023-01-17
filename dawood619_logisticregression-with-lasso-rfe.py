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

import warnings

warnings.filterwarnings("ignore")
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

samplesub_df = pd.read_csv('../input/sample_submission.csv')
train_df.head()
test_df.head()
samplesub_df.head()
print(train_df.shape, test_df.shape,samplesub_df.shape)
print(train_df['target'].unique())
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(train_df['target'])
plt.figure(figsize=[8,8])

train_df['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)

plt.show()
print(train_df['target'].value_counts())

Total = np.add(train_df['target'].where(train_df['target']==1).value_counts(), 

               train_df['target'].where(train_df['target']==0).value_counts())

print('target 1 : ',(train_df['target'].where(train_df['target']==1).value_counts()/Total)*100)

print('target 0 : ',100 - (train_df['target'].where(train_df['target']==1).value_counts()/Total)*100)
print(train_df.isnull().any().sum())

print(test_df.isnull().any().sum())
train_df.describe()
cols = ['target','id']

X = train_df.drop(cols,axis=1)

y = train_df['target']

X_test = test_df.drop('id',axis=1)
from sklearn.model_selection import cross_val_score



#defining a generic Function to give ROC_AUC Scores in table format for better readability

def crossvalscore(model):

    scores = cross_val_score(model,X,y,cv=5,scoring='roc_auc',n_jobs=-1)

    acc = cross_val_score(model,X,y,cv=5,scoring='accuracy',n_jobs=-1)

    rand_scores = pd.DataFrame({

    'cv':range(1,6),

    'roc_auc score':scores,

    'accuracy score':acc

    })

    print('AUC :',rand_scores['roc_auc score'].mean())

    print('accuracy :',rand_scores['accuracy score'].mean())

    return rand_scores.sort_values(by='roc_auc score',ascending=False)
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(max_depth=2,random_state=42)

crossvalscore(rand_clf)
#logistic regression using PCA

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

pca = PCA(n_components=42)

X_new = pca.fit_transform(X)



log_clf = LogisticRegression(C = 0.1, class_weight= 'balanced', penalty= 'l1', solver= 'liblinear',random_state=42)

scores = cross_val_score(log_clf,X_new,y,cv=5,scoring='roc_auc',n_jobs=-1)

acc = cross_val_score(log_clf,X_new,y,cv=5,scoring='accuracy',n_jobs=-1)

rand_scores = pd.DataFrame({

'cv':range(1,6),

'roc_auc score':scores,

'accuracy score':acc

})

print('AUC :',rand_scores['roc_auc score'].mean())

print('accuracy :',rand_scores['accuracy score'].mean())

rand_scores.sort_values(by='roc_auc score',ascending=False)
from sklearn.linear_model import LogisticRegression

#simple logistic regression with lasso regularization

log_clf = LogisticRegression(C = 0.1, class_weight= 'balanced', penalty= 'l1', solver= 'liblinear',random_state=42)

crossvalscore(log_clf)
from sklearn.feature_selection import RFE

log_clf = LogisticRegression(C = 0.1, class_weight= 'balanced', penalty= 'l1', solver= 'liblinear',random_state=42)

selector = RFE(log_clf,21)

selector.fit(X,y)

crossvalscore(selector)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss='log',penalty='elasticnet')

crossvalscore(sgd_clf)
from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import VotingClassifier,BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



log_clf = LogisticRegression(C = 0.1, class_weight= 'balanced', penalty= 'l1', solver= 'liblinear',random_state=42)

log_selector = RFE(log_clf,21)

log_selector.fit(X,y)

rand_clf = RandomForestClassifier(max_depth=2,random_state=42)

svm_clf = SVC(gamma='auto',probability=True, random_state=42)

extra_clf = ExtraTreesClassifier(max_depth=2,random_state=42)

#ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,algorithm='SAMME.R', learning_rate=0.5, random_state=42)



voting_clf = VotingClassifier(

    estimators = [('lr',log_selector),('rf',rand_clf),('ex',extra_clf),('sv',svm_clf)],

    voting='soft')

crossvalscore(selector)
voting_clf.fit(X,y)



#submission = pd.read_csv('../input/sample_submission.csv')

#submission['target'] = voting_clf.predict_proba(X_test)

#submission.to_csv('submission1.csv', index=False)