# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from __future__ import division, print_function

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split

import time
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.describe()
# Missing value

df.isna().sum().sum()
df.Class.value_counts()
df.Class.value_counts().plot(kind='bar', figsize=(10, 10));
plt.figure(figsize=(12, 8))

sns.distplot(df.Time);
plt.figure(figsize=(12, 8))

sns.distplot(df.V28);
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

sns.distplot(df.Time, ax=ax1)

sns.distplot(df.Amount, ax=ax2)

plt.tight_layout()

plt.savefig('Time_Amount_dist.png');
n_cols = 5

n_rows = 6

fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))



for i in range(df.shape[1] - 1):

    sns.boxplot(df.Class, df.iloc[:, i], ax=axes[i // n_cols, i % n_cols])

    

plt.savefig('boxplot.png');
n_cols = 5

n_rows = 6

fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))



for i in range(df.shape[1] - 1):

    sns.distplot(df.iloc[:, i], ax=axes[i // n_cols, i % n_cols])

    

plt.savefig('distplot.png');
#sns.pairplot(df, hue='Class');
seed = 42

train_df, test_df = train_test_split(df, test_size=0.1, random_state=seed)

train_df, val_df = train_test_split(train_df, test_size=1/9, random_state=seed)
train_df.Class.value_counts()
val_df.Class.value_counts()
test_df.Class.value_counts()
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
train_df_copy = train_df.copy()

val_df_copy = val_df.copy()

test_df_copy = test_df.copy()
sc_time = StandardScaler()

sc_amount = StandardScaler()

train_df_copy.loc[:, 'Time'] = sc_time.fit_transform(train_df_copy.loc[:, ['Time']]).ravel()

train_df_copy.loc[:, 'Amount'] = sc_amount.fit_transform(train_df_copy.loc[:, ['Amount']]).ravel()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

sns.distplot(train_df_copy.Time, ax=ax1)

sns.distplot(train_df_copy.Amount, ax=ax2)

plt.tight_layout()

plt.savefig('Time_Amount_scaled_dist.png');
n_cols = 5

n_rows = 6

fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))



for i in range(train_df_copy.shape[1] - 1):

    sns.boxplot(train_df_copy.Class,

                train_df_copy.iloc[:, i],

                ax=axes[i // n_cols, i % n_cols])

    

plt.savefig('train_boxplot.png');
n_cols = 5

n_rows = 6

fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))



for i in range(train_df_copy.shape[1] - 1):

    sns.distplot(train_df_copy.iloc[:, i], ax=axes[i // n_cols, i % n_cols])

    

plt.savefig('train_distplot.png');
val_df_copy.loc[:, 'Time'] = sc_time.transform(val_df_copy.loc[:, ['Time']]).ravel()

val_df_copy.loc[:, 'Amount'] = sc_amount.transform(val_df_copy.loc[:, ['Amount']]).ravel()
test_df_copy.loc[:, 'Time'] = sc_time.transform(test_df_copy.loc[:, ['Time']]).ravel()

test_df_copy.loc[:, 'Amount'] = sc_amount.transform(test_df_copy.loc[:, ['Amount']]).ravel()
X_train, y_train = train_df_copy.drop('Class', axis=1).values, train_df_copy.Class.values

X_val, y_val = val_df_copy.drop('Class', axis=1).values, val_df_copy.Class.values

X_test, y_test = test_df_copy.drop('Class', axis=1).values, test_df_copy.Class.values
seed = 42



classifiers = [

    ('Logistic regression', LogisticRegression(random_state=seed)),

    ('SVM', SVC(random_state=seed)),

    ('KNN', KNeighborsClassifier(n_neighbors=10)),

    ('Decision tree', DecisionTreeClassifier(random_state=seed)),

    ('Random forest', RandomForestClassifier(n_estimators=100, random_state=seed))

]
# Code for cross validation, but take long time



# cv_summary = []

# test_summary = []

# test_predictions = []

# test_prediction_probs = []



# for name, classifier in classifiers:

#     print('Model: {}'.format(name))

    

#     record = {'Model': name}

#     scores = cross_validate(classifier, X_train, y_train, scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))

#     record['Accuracy'] = scores['test_accuracy'].mean()

#     record['Precision'] = scores['test_precision'].mean()

#     record['Recall'] = scores['test_recall'].mean()

#     record['F1-score'] = scores['test_f1'].mean()

#     record['AUC'] = scores['test_roc_auc'].mean()

#     cv_summary.append(record)



#     classifier.fit(X_train, y_train)

#     y_pred = classifier.predict(X_test)

#     test_predictions.append(y_test)

#     if name == 'SVM':

#         y_prob = classifier.decision_function(X_test)

#         test_prediction_probs.append(y_prob)

#     else:

#         y_prob = classifier.predict_proba(X_test)

#         test_prediction_probs.append(y_prob[:, 1])



#     cm = confusion_matrix(y_test, y_pred)

#     print('Confusion matrix')

#     print(cm)



#     record = {'Model': name}

#     record['Accuracy'] = accuracy_score(y_test, y_pred)

#     record['Precision'] = precision_score(y_test, y_pred)

#     record['Recall'] = recall_score(y_test, y_pred)

#     record['F1-score'] = f1_score(y_test, y_pred)

#     record['AUC'] = roc_auc_score(y_test, test_prediction_probs[-1])

#     print('-' * 20)

#     test_summary.append(record)
# pd.DataFrame(cv_summary)
# pd.DataFrame(test_summary)
val_summary = []

test_summary = []

test_predictions = []

test_prediction_probs = []



for name, classifier in classifiers:

    print('Model: {}'.format(name))

    start_time = time.time()

    classifier.fit(X_train, y_train)

    end_time = time.time()

    print('Training time: {:.2f}'.format(end_time - start_time))

    # Validation set

    y_pred = classifier.predict(X_val)

    if name == 'SVM':

        y_prob = classifier.decision_function(X_val)

    else:

        y_prob = classifier.predict_proba(X_val)

        y_prob = y_prob[:, 1]

    

    record = {'Model': name}

    record['Accuracy'] = accuracy_score(y_val, y_pred)

    record['Precision'] = precision_score(y_val, y_pred)

    record['Recall'] = recall_score(y_val, y_pred)

    record['F1-score'] = f1_score(y_val, y_pred)

    record['AUC'] = roc_auc_score(y_val, y_prob)

    val_summary.append(record)



    # Test set

    start_time = time.time()

    y_pred = classifier.predict(X_test)

    end_time = time.time()

    print('Prediction time: {:.2f}'.format(end_time - start_time))

    

    test_predictions.append(y_test)

    if name == 'SVM':

        y_prob = classifier.decision_function(X_test)

        test_prediction_probs.append(y_prob)

    else:

        y_prob = classifier.predict_proba(X_test)

        test_prediction_probs.append(y_prob[:, 1])



    cm = confusion_matrix(y_test, y_pred)

    print('Confusion matrix')

    print(cm)



    record = {'Model': name}

    record['Accuracy'] = accuracy_score(y_test, y_pred)

    record['Precision'] = precision_score(y_test, y_pred)

    record['Recall'] = recall_score(y_test, y_pred)

    record['F1-score'] = f1_score(y_test, y_pred)

    record['AUC'] = roc_auc_score(y_test, test_prediction_probs[-1])

    print('-' * 20)

    test_summary.append(record)
pd.DataFrame(val_summary)
pd.DataFrame(test_summary)
fpr = dict()

tpr = dict()

auc = dict()

for i, (name, _) in enumerate(classifiers):

    fpr[name], tpr[name], _ = roc_curve(y_test, test_prediction_probs[i])

    auc[name] = roc_auc_score(y_test, test_prediction_probs[i])
plt.figure(figsize=(12, 8))

lw = 2

for name in fpr.keys():

    plt.plot(fpr[name], tpr[name], lw=lw, label='ROC curve of {} (area = {:.4f})'.format(name, auc[name]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('ROC_full.png');
train_df_copy = train_df.copy()

val_df_copy = val_df.copy()

test_df_copy = test_df.copy()
from imblearn.under_sampling import RandomUnderSampler
X_train, y_train = train_df_copy.drop('Class', axis=1), train_df_copy.Class

X_val, y_val = val_df_copy.drop('Class', axis=1), val_df_copy.Class

X_test, y_test = test_df_copy.drop('Class', axis=1), test_df_copy.Class



seed = 42

X_train, y_train = RandomUnderSampler(random_state=seed).fit_sample(X_train, y_train)



sc_time = StandardScaler()

sc_amount = StandardScaler()

X_train.loc[:, 'Time'] = sc_time.fit_transform(X_train.loc[:, ['Time']]).ravel()

X_train.loc[:, 'Amount'] = sc_amount.fit_transform(X_train.loc[:, ['Amount']]).ravel()
X_val.loc[:, 'Time'] = sc_time.fit_transform(X_val.loc[:, ['Time']]).ravel()

X_val.loc[:, 'Amount'] = sc_amount.fit_transform(X_val.loc[:, ['Amount']]).ravel()

X_test.loc[:, 'Time'] = sc_time.transform(X_test.loc[:, ['Time']]).ravel()

X_test.loc[:, 'Amount'] = sc_amount.transform(X_test.loc[:, ['Amount']]).ravel()
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
X_pca = PCA(n_components=2).fit_transform(X_train)

plt.figure(figsize=(12, 8))

for i in y_train.value_counts().index:

    plt.scatter(X_pca[y_train==i, 0], X_pca[y_train==i, 1], label='Class {}'.format(i))

plt.legend()

plt.xlabel('1st component')

plt.ylabel('2nd component')

plt.savefig('PCA_under_sampling.png');
X_tsne = TSNE(n_components=2).fit_transform(X_train)

plt.figure(figsize=(12, 8))

for i in y_train.value_counts().index:

    plt.scatter(X_tsne[y_train==i, 0], X_tsne[y_train==i, 1], label='Class {}'.format(i))

plt.legend()

plt.xlabel('1st component')

plt.ylabel('2nd component')

plt.savefig('TSNE_under_sampling.png');
X_train, y_train = X_train.values, y_train.values

X_val, y_val = X_val.values, y_val.values

X_test, y_test = X_test.values, y_test.values
seed = 42



classifiers = [

    ('Logistic regression', LogisticRegression(random_state=seed)),

    ('SVM', SVC(random_state=seed)),

    ('KNN', KNeighborsClassifier(n_neighbors=10)),

    ('Decision tree', DecisionTreeClassifier(random_state=seed)),

    ('Random forest', RandomForestClassifier(n_estimators=100, random_state=seed))

]
val_summary = []

test_summary = []

test_predictions = []

test_prediction_probs = []



for name, classifier in classifiers:

    print('Model: {}'.format(name))

    start_time = time.time()

    classifier.fit(X_train, y_train)

    end_time = time.time()

    print('Training time: {:.2f}'.format(end_time - start_time))

    

    # Validation set

    y_pred = classifier.predict(X_val)

    if name == 'SVM':

        y_prob = classifier.decision_function(X_val)

    else:

        y_prob = classifier.predict_proba(X_val)

        y_prob = y_prob[:, 1]

    

    record = {'Model': name}

    record['Accuracy'] = accuracy_score(y_val, y_pred)

    record['Precision'] = precision_score(y_val, y_pred)

    record['Recall'] = recall_score(y_val, y_pred)

    record['F1-score'] = f1_score(y_val, y_pred)

    record['AUC'] = roc_auc_score(y_val, y_prob)

    val_summary.append(record)



    # Test set

    start_time = time.time()

    y_pred = classifier.predict(X_test)

    end_time = time.time()

    print('Training time: {:.2f}'.format(end_time - start_time))

    test_predictions.append(y_test)

    if name == 'SVM':

        y_prob = classifier.decision_function(X_test)

        test_prediction_probs.append(y_prob)

    else:

        y_prob = classifier.predict_proba(X_test)

        test_prediction_probs.append(y_prob[:, 1])



    cm = confusion_matrix(y_test, y_pred)

    print('Confusion matrix')

    print(cm)



    record = {'Model': name}

    record['Accuracy'] = accuracy_score(y_test, y_pred)

    record['Precision'] = precision_score(y_test, y_pred)

    record['Recall'] = recall_score(y_test, y_pred)

    record['F1-score'] = f1_score(y_test, y_pred)

    record['AUC'] = roc_auc_score(y_test, test_prediction_probs[-1])

    print('-' * 20)

    test_summary.append(record)
pd.DataFrame(val_summary)
pd.DataFrame(test_summary)
fpr = dict()

tpr = dict()

auc = dict()

for i, (name, _) in enumerate(classifiers):

    fpr[name], tpr[name], _ = roc_curve(y_test, test_prediction_probs[i])

    auc[name] = roc_auc_score(y_test, test_prediction_probs[i])



plt.figure(figsize=(12, 8))

lw = 2

for name in fpr.keys():

    plt.plot(fpr[name], tpr[name], lw=lw, label='ROC curve of {} (area = {:.4f})'.format(name, auc[name]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('ROC_under_sampling.png');
from imblearn.over_sampling import SMOTE
X_train, y_train = train_df_copy.drop('Class', axis=1), train_df_copy.Class

X_val, y_val = val_df_copy.drop('Class', axis=1), val_df_copy.Class

X_test, y_test = test_df_copy.drop('Class', axis=1), test_df_copy.Class



seed = 42

X_train, y_train = SMOTE(random_state=seed).fit_sample(X_train, y_train)



sc_time = StandardScaler()

sc_amount = StandardScaler()

X_train.loc[:, 'Time'] = sc_time.fit_transform(X_train.loc[:, ['Time']]).ravel()

X_train.loc[:, 'Amount'] = sc_amount.fit_transform(X_train.loc[:, ['Amount']]).ravel()
X_val.loc[:, 'Time'] = sc_time.fit_transform(X_val.loc[:, ['Time']]).ravel()

X_val.loc[:, 'Amount'] = sc_amount.fit_transform(X_val.loc[:, ['Amount']]).ravel()

X_test.loc[:, 'Time'] = sc_time.transform(X_test.loc[:, ['Time']]).ravel()

X_test.loc[:, 'Amount'] = sc_amount.transform(X_test.loc[:, ['Amount']]).ravel()
X_train, y_train = X_train.values, y_train.values

X_val, y_val = X_val.values, y_val.values

X_test, y_test = X_test.values, y_test.values
seed = 42



classifiers = [

    ('Logistic regression', LogisticRegression(random_state=seed)),

#     ('SVM', SVC(random_state=seed)),

#     ('KNN', KNeighborsClassifier(n_neighbors=10)),

    ('Decision tree', DecisionTreeClassifier(random_state=seed)),

    ('Random forest', RandomForestClassifier(n_estimators=100, random_state=seed))

]
val_summary = []

test_summary = []

test_predictions = []

test_prediction_probs = []



for name, classifier in classifiers:

    print('Model: {}'.format(name))



    start_time = time.time()

    classifier.fit(X_train, y_train)

    end_time = time.time()

    print('Training time: {:.2f}'.format(end_time - start_time))

    

    # Validation set

    y_pred = classifier.predict(X_val)

    if name == 'SVM':

        y_prob = classifier.decision_function(X_val)

    else:

        y_prob = classifier.predict_proba(X_val)

        y_prob = y_prob[:, 1]

    

    record = {'Model': name}

    record['Accuracy'] = accuracy_score(y_val, y_pred)

    record['Precision'] = precision_score(y_val, y_pred)

    record['Recall'] = recall_score(y_val, y_pred)

    record['F1-score'] = f1_score(y_val, y_pred)

    record['AUC'] = roc_auc_score(y_val, y_prob)

    val_summary.append(record)



    # Test set

    start_time = time.time()

    y_pred = classifier.predict(X_test)

    end_time = time.time()

    print('Training time: {:.2f}'.format(end_time - start_time))

    

    test_predictions.append(y_test)

    if name == 'SVM':

        y_prob = classifier.decision_function(X_test)

        test_prediction_probs.append(y_prob)

    else:

        y_prob = classifier.predict_proba(X_test)

        test_prediction_probs.append(y_prob[:, 1])



    cm = confusion_matrix(y_test, y_pred)

    print('Confusion matrix')

    print(cm)



    record = {'Model': name}

    record['Accuracy'] = accuracy_score(y_test, y_pred)

    record['Precision'] = precision_score(y_test, y_pred)

    record['Recall'] = recall_score(y_test, y_pred)

    record['F1-score'] = f1_score(y_test, y_pred)

    record['AUC'] = roc_auc_score(y_test, test_prediction_probs[-1])

    print('-' * 20)

    test_summary.append(record)
pd.DataFrame(val_summary)
pd.DataFrame(test_summary)
fpr = dict()

tpr = dict()

auc = dict()

for i, (name, _) in enumerate(classifiers):

    fpr[name], tpr[name], _ = roc_curve(y_test, test_prediction_probs[i])

    auc[name] = roc_auc_score(y_test, test_prediction_probs[i])



plt.figure(figsize=(12, 8))

lw = 2

for name in fpr.keys():

    plt.plot(fpr[name], tpr[name], lw=lw, label='ROC curve of {} (area = {:.4f})'.format(name, auc[name]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.savefig('ROC_over_sampling.png');