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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv("../input/minor-project-2020/train.csv")
test_df = pd.read_csv("../input/minor-project-2020/test.csv")
train_df.describe()
test_df.describe()
count_target = pd.value_counts(train_df['target'], sort=True).sort_index()
count_target
# minority: 1s
no_one_target = len(train_df[train_df.target == 1])
one_indices = np.array(train_df[train_df.target == 1].index)

# majority: 0s
zero_indices = np.array(train_df[train_df.target == 0].index)

# undersample
reduced_zero_indices = np.random.choice(zero_indices, 1 * no_one_target, replace=False)
reduced_zero_indices = np.array(reduced_zero_indices)

under_sample_indices = np.concatenate([one_indices, reduced_zero_indices])
under_sample_df = train_df.iloc[under_sample_indices, :]

under_sample_df.describe()
# X_train = train_df.drop(['id', 'target'], axis=1)
# Y_train = train_df['target']

X_train_undersample = under_sample_df.drop(['id', 'target'], axis=1)
Y_train_undersample = under_sample_df['target']

X_test = test_df.drop('id', axis=1).copy()
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scaled_X_train = scalar.fit_transform(X_train_undersample)
scaled_X_test = scalar.transform(X_test)
rf = RandomForestClassifier(n_estimators=50)
rf.fit(scaled_X_train, Y_train_undersample)
Y_pred = rf.predict(scaled_X_test)
Y_pred_prob = rf.predict_proba(scaled_X_test)

rf.score(scaled_X_train, Y_train_undersample)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 


Y_pred
Y_submission = Y_pred_prob[:, 1]
submission = pd.DataFrame({
        'id': test_df['id'],
        'target': Y_submission
    })
submission.to_csv('submission_rfc_scaled.csv', index=False)
count_target = pd.value_counts(Y_train_undersample, sort=True).sort_index()
count_target
Y_pred2 = rf.predict(scaled_X_train)
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

conf_mat = confusion_matrix(y_true=Y_train_undersample, y_pred=Y_pred2)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
train_df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(under_sample_df.drop(columns = ['target', 'id']), under_sample_df.target, test_size = 0.2 , random_state = 42)
etc = ExtraTreesClassifier(n_estimators = 1000, random_state = 42)

etc.fit(scaled_X_train, Y_train_undersample)
Y_pred = etc.predict(scaled_X_test)

etc.score(scaled_X_train, Y_train_undersample)
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()

count_target = pd.value_counts(y_train, sort=True).sort_index()
count_target
etc = ExtraTreesClassifier(n_estimators=1000, random_state = 42)
etc.fit(X_train, y_train)
etc.score(X_test, y_test)
rfc = RandomForestClassifier(n_estimators=1000, random_state = 42)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
y_pred = etc.predict(X_test)
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
etc.fit(under_sample_df.drop(columns=['target', 'id']), under_sample_df.target)
etc.score(under_sample_df.drop(columns=['target', 'id']), under_sample_df.target)
submission = pd.DataFrame({
        'id': test_df['id'],
        'target': etc.predict(test_df.drop(columns = 'id'))
    })
submission.to_csv('submission_scaled.csv', index=False)

