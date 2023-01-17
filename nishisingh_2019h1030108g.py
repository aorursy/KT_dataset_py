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
train_df=pd.read_csv('/kaggle/input/minor-project-2020/train.csv')

test_df=pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
train_df.head()
test_df.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_curve, auc, log_loss

X= train_df.drop('target',axis=1)

Y= train_df['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=22)



lr = LogisticRegression()

model=lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

Y_proba = lr.predict_proba(X_test)[:, 1]

[fpr, tpr, thr] = roc_curve(Y_test, Y_proba)

print(" accuracy is %2.4f" % accuracy_score(Y_test, Y_pred))

print(" log_loss is %2.4f" % log_loss(Y_test, Y_proba))

print(" auc is %2.4f" % auc(fpr, tpr))

result=test_df

result['id']=test_df['id']

result['target']=model.predict(test_df)

submission=result[['id','target']]

submission.to_csv("submission1.csv",index=False)
submission.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_curve, auc, log_loss

from imblearn.under_sampling import RandomUnderSampler

X= train_df.drop('target',axis=1)

Y= train_df['target']

undersample = RandomUnderSampler(sampling_strategy='majority')

# fit and apply the transform

X_over, Y_over = undersample.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_over, Y_over, test_size=0.2, random_state=22)



lr = LogisticRegression()

model=lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

Y_proba = lr.predict_proba(X_test)[:, 1]

[fpr, tpr, thr] = roc_curve(Y_test, Y_proba)

print(" accuracy is %2.4f" % accuracy_score(Y_test, Y_pred))

print(" log_loss is %2.4f" % log_loss(Y_test, Y_proba))

print(" auc is %2.4f" % auc(fpr, tpr))



test_df=pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

result=test_df

result['id']=test_df['id']

result['target']=model.predict(test_df)

submission=result[['id','target']]

submission.to_csv("submission2.csv",index=False)
# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import KFold

# from sklearn.model_selection import LeaveOneOut

# from sklearn.model_selection import LeavePOut

# from sklearn.model_selection import ShuffleSplit

# from sklearn.model_selection import StratifiedKFold

# from sklearn import model_selection

# x1=train_df.drop('target',axis=1).values

# y1=train_df['target'].values

# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)

# model = LogisticRegression()

# model.fit(X_train, Y_train)

# result = model.score(X_test, Y_test)

# print("Accuracy: %.2f%%" % (result*100.0))

# kfold = model_selection.KFold(n_splits=10, random_state=100)

# model_kfold = LogisticRegression()

# results_kfold = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold)

# print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 

# skfold = StratifiedKFold(n_splits=3, random_state=100)

# model_skfold = LogisticRegression()

# results_skfold = model_selection.cross_val_score(model_skfold, x1, y1, cv=skfold)

# print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))
# test_df=pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

# result=test_df

# result['id']=test_df['id']

# result['target']=model.predict(test_df)

# submission=result[['id','target']]

# submission.to_csv("submission5.csv",index=False)

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import accuracy_score, roc_curve, auc, log_loss

# from imblearn.over_sampling import RandomOverSampler

# X= train_df.drop('target',axis=1)

# Y= train_df['target']

# oversample = RandomOverSampler(sampling_strategy='minority')

# # fit and apply the transform

# X_over, Y_over = oversample.fit_resample(X, Y)

# X_train, X_test, Y_train, Y_test = train_test_split(X_over, Y_over, test_size=0.2, random_state=22)



# lr = LogisticRegression()

# model=lr.fit(X_train, Y_train)

# Y_pred = lr.predict(X_test)

# Y_proba = lr.predict_proba(X_test)[:, 1]

# [fpr, tpr, thr] = roc_curve(Y_test, Y_proba)

# print(" accuracy is %2.4f" % accuracy_score(Y_test, Y_pred))

# print(" log_loss is %2.4f" % log_loss(Y_test, Y_proba))

# print(" auc is %2.4f" % auc(fpr, tpr))
# test_df=pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

# result=test_df

# result['id']=test_df['id']

# result['target']=model.predict(test_df)

# submission=result[['id','target']]

# submission.to_csv("submission3.csv",index=False)
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import accuracy_score, roc_curve, auc, log_loss

# from imblearn.under_sampling import RandomUnderSampler

# X= train_df.drop('target',axis=1)

# Y= train_df['target']

# #undersample = RandomUnderSampler(sampling_strategy='majority')

# # fit and apply the transform

# X_over, Y_over = undersample.fit_resample(X, Y)

# X_train, X_test, Y_train, Y_test = train_test_split(X_over, Y_over, test_size=0.2, random_state=22)



# lr = RandomForestClassifier()

# model=lr.fit(X_train, Y_train)

# Y_pred = lr.predict(X_test)

# Y_proba = lr.predict_proba(X_test)[:, 1]

# [fpr, tpr, thr] = roc_curve(Y_test, Y_proba)

# print(" accuracy is %2.4f" % accuracy_score(Y_test, Y_pred))

# print(" log_loss is %2.4f" % log_loss(Y_test, Y_proba))

# print(" auc is %2.4f" % auc(fpr, tpr))



# test_df=pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

# result=test_df

# result['id']=test_df['id']

# result['target']=model.predict(test_df)

# submission=result[['id','target']]

# submission.to_csv("submission4.csv",index=False)
# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import accuracy_score, roc_curve, auc, log_loss

# from imblearn.under_sampling import RandomUnderSampler

# from imblearn.over_sampling import RandomOverSampler

# X= train_df.drop('target',axis=1)

# Y= train_df['target']

# over = RandomOverSampler(sampling_strategy='minority')

# # fit and apply the transform

# X_over, Y_over = over.fit_resample(X, Y)

# under= RandomUnderSampler(sampling_strategy='majority')

# X_under, Y_under = under.fit_resample(X_over, Y_over)

# X_train, X_test, Y_train, Y_test = train_test_split(X_under, Y_under, test_size=0.2, random_state=22)



# lr = LogisticRegression()

# model=lr.fit(X_train, Y_train)

# Y_pred = lr.predict(X_test)

# Y_proba = lr.predict_proba(X_test)[:, 1]

# [fpr, tpr, thr] = roc_curve(Y_test, Y_proba)

# print(" accuracy is %2.4f" % accuracy_score(Y_test, Y_pred))

# print(" log_loss is %2.4f" % log_loss(Y_test, Y_proba))

# print(" auc is %2.4f" % auc(fpr, tpr))



#  test_df=pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

# result=test_df

# result['id']=test_df['id']

# result['target']=model.predict(test_df)

# submission=result[['id','target']]

# submission.to_csv("submission4.csv",index=False)
# logreg = LogisticRegression()

# scores_accuracy = cross_val_score(logreg, X, Y, cv=10, scoring='accuracy')

# scores_log_loss = cross_val_score(logreg, X, Y, cv=10, scoring='neg_log_loss')

# scores_auc = cross_val_score(logreg, X, Y, cv=10, scoring='roc_auc')

# print('K-fold cross-validation results:')

# print(logreg.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())

# print(logreg.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())

# print(logreg.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())