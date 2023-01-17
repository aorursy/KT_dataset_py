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
print(os.listdir("../input/exams/Exams"))

train = pd.read_csv("../input/exams/Exams/train.gz")

test = pd.read_csv("../input/exams/Exams/test.gz")

y_test = pd.read_csv("../input/exams/Exams/y_test.gz")
train.head()
test.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(17,10))

sns.countplot(x="click",data=train)

print(train.click.value_counts())

print(53649 / train.shape[0])
train.info()
len(train['0'].unique())
len(train['1'].unique())
fig, axs = plt.subplots(ncols=2,nrows=2,figsize=(17,10))

sns.countplot(x="2", hue="click", data=train, ax=axs[0][0])

sns.countplot(x="3", hue="click", data=train, ax=axs[0][1])

sns.countplot(x="4", hue="click", data=train, ax=axs[1][0])

sns.countplot(x="6", hue="click", data=train, ax=axs[1][1])
plt.figure(figsize=(17,10))

sns.countplot(x="5", hue="click", data=train)
for i in ['7','8','9','10']:

    print('number of unique of col %s: %s' %(i,len(train[i].unique())))
# t = [str(i) for i in range(11,34)]

# sns.pairplot(train[t])
len(set(test['0'].unique()) - set(train['0'].unique()))
len(set(test['0']))
len(set(test['1'].unique()) - set(train['1'].unique()))
len(set(test['1']))
len(set(train['1']))
y_test.click.value_counts()
y = train.click

train = train.drop('click',axis=1)
from sklearn.preprocessing import LabelEncoder

for f in train.columns:

    if train[f].dtype=='object' or test[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(train[f].values) + list(test[f].values))

        train[f] = lbl.transform(list(train[f].values))

        test[f] = lbl.transform(list(test[f].values)) 
# num_df_train = train.select_dtypes(include=['float64'])
# num_df_test = test.select_dtypes(include=['float64'])
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score,recall_score, precision_score, accuracy_score,roc_curve,auc, classification_report

X = train.values

X_test = test.values

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)
def metrics(true, preds):

    print(confusion_matrix(true, preds))

    accuracy = accuracy_score(true, preds)

    recall = recall_score(true, preds)

    precision = precision_score(true, preds)

    f1score = f1_score(true, preds)

    print ('accuracy: {}, recall: {}, precision: {}, f1-score: {}'.format(accuracy, recall, precision, f1score))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_val, y_val)

y_pred = model.predict(X_val)

print(classification_report(y_val,y_pred))
from imblearn.over_sampling import (RandomOverSampler, 

                                    SMOTE, 

                                    ADASYN)

sampler = SMOTE(sampling_strategy='minority')

X_s, y_s = sampler.fit_sample(X_train,y_train)
logreg = LogisticRegression()

logreg.fit(X_s, y_s)

y_pred = logreg.predict(X_val)

print(classification_report(y_val,y_pred))
y_pred_prob = logreg.predict_proba(X_val)

plt.figure()

fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob[:,1])

roc_auc = auc(fpr, tpr) # compute area under the curve

plt.figure(figsize=(17,10))

plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % (roc_auc))

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

ax2 = plt.gca().twinx()

ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')

ax2.set_ylabel('Threshold',color='r')

ax2.set_ylim([thresholds[-1],thresholds[0]])

ax2.set_xlim([fpr[0],fpr[-1]])
import lightgbm as lgb
params = {

    'objective' :'binary',

    'learning_rate' : 0.02,

    'num_leaves' : 144,

    'feature_fraction': 0.64, 

    'bagging_fraction': 0.8, 

    'bagging_freq':1,

    'boosting_type' : 'gbdt',

    'metric': 'binary_logloss'

}
d_train = lgb.Dataset(X_s, y_s)

d_val = lgb.Dataset(X_val, y_val, reference=d_train)

lgb_clf = lgb.train(train_set=d_train,

                    valid_sets=[d_val],

                    params=params,

                    num_boost_round=1000,

                    early_stopping_rounds=10,

                    verbose_eval=False)
y_pred_prob = lgb_clf.predict(X_val)

y_pred = [1 if x > 0.5 else 0 for x in y_pred_prob]

print(classification_report(y_val,y_pred))
plt.figure()

fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)

roc_auc = auc(fpr, tpr) # compute area under the curve

plt.figure(figsize=(17,10))

plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % (roc_auc))

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

ax2 = plt.gca().twinx()

ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')

ax2.set_ylabel('Threshold',color='r')

ax2.set_ylim([thresholds[-1],thresholds[0]])

ax2.set_xlim([fpr[0],fpr[-1]])
y_hat_prob = lgb_clf.predict(X_test)

plt.figure()

fpr, tpr, thresholds = roc_curve(y_test.click, y_hat_prob)

roc_auc = auc(fpr, tpr) # compute area under the curve

plt.figure(figsize=(17,10))

plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % (roc_auc))

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

ax2 = plt.gca().twinx()

ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')

ax2.set_ylabel('Threshold',color='r')

ax2.set_ylim([thresholds[-1],thresholds[0]])

ax2.set_xlim([fpr[0],fpr[-1]])
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=20, random_state=42)

rf_clf.fit(X_s, y_s)

y_pred = rf_clf.predict(X_val)

metrics(y_val,y_pred)
y_pred_prob = rf_clf.predict_proba(X_val)

plt.figure()

fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob[:,1])

roc_auc = auc(fpr, tpr) # compute area under the curve

plt.figure(figsize=(17,10))

plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % (roc_auc))

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

ax2 = plt.gca().twinx()

ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')

ax2.set_ylabel('Threshold',color='r')

ax2.set_ylim([thresholds[-1],thresholds[0]])

ax2.set_xlim([fpr[0],fpr[-1]])
import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators=1000,

                        n_jobs=4,

                        max_depth=9,

                        learning_rate=0.05,

                        subsample=0.9,

                        colsample_bytree=0.9,

                        missing=-999,

                        tree_method='gpu_hist')



%time clf.fit(X_s, y_s)
y_pred_prob = clf.predict_proba(X_val)
plt.figure()

fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob[:,1])

roc_auc = auc(fpr, tpr) # compute area under the curve

plt.figure(figsize=(17,10))

plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % (roc_auc))

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

ax2 = plt.gca().twinx()

ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')

ax2.set_ylabel('Threshold',color='r')

ax2.set_ylim([thresholds[-1],thresholds[0]])

ax2.set_xlim([fpr[0],fpr[-1]])
y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))
y_hat = clf.predict(X_test)

y_hat_prob = clf.predict_proba(X_test)

print(classification_report(y_test.click,y_hat))
y_hat_prob = clf.predict_proba(X_test)

plt.figure()

fpr, tpr, thresholds = roc_curve(y_test.click, y_hat_prob[:,1])

roc_auc = auc(fpr, tpr) # compute area under the curve

plt.figure(figsize=(17,10))

plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % (roc_auc))

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

ax2 = plt.gca().twinx()

ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')

ax2.set_ylabel('Threshold',color='r')

ax2.set_ylim([thresholds[-1],thresholds[0]])

ax2.set_xlim([fpr[0],fpr[-1]])