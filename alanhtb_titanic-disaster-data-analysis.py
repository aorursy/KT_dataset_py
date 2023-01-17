# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib notebook

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.describe()
train.isnull().sum()
train_num = train[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
train_num
for i in range(len(train_num)):
    train_num['Sex'][i] = 1 if train_num['Sex'][i]=='male' else 0

train_num
train_num.Age.fillna(value=train_num.Age.median(), inplace=True)
train_cat = train[['Embarked']]
train_cat
train_cat.Embarked.fillna(value=train_cat.Embarked.mode()[0], inplace=True)
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
train_cat_1hot = cat_encoder.fit_transform(train_cat)
train_cat_1hot
cat_encoder.categories_
train_cat_ = pd.DataFrame(train_cat_1hot.toarray(), columns=['C', 'Q', 'S'])
train_cat_
X = pd.concat([train_num, train_cat_], axis=1)
X
y = train['Survived']
pd.concat([y, X], axis=1).corr()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=18)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[('log', log_clf), ('rnd', rnd_clf), ('svm', svm_clf)],
    voting='hard')
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(clf.__class__.__name__, accuracy_score(y_val, y_pred))
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)

voting_clf = VotingClassifier(
    estimators=[('log', log_clf), ('rnd', rnd_clf), ('svm', svm_clf)],
    voting='soft')

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(clf.__class__.__name__, accuracy_score(y_val, y_pred))
test_num = test[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
test_num
test_num.Sex[test_num.Sex=='male'] = 1
test_num.Sex[test_num.Sex=='female'] = 0
test_num
test_num.Age.fillna(value=test_num.Age.median(), inplace=True)
test_num.Fare.fillna(value=test_num.Fare.median(), inplace=True)
test_num
test_cat = test[['Embarked']]
test_cat_1hot = cat_encoder.fit_transform(test_cat)
test_cat_1hot.toarray()
test_cat_ = pd.DataFrame(test_cat_1hot.toarray(), columns=['C', 'Q', 'S'])
test_cat_
X_test = pd.concat([test_num, test_cat_], axis=1)
X_test
X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled
y_test_pred = voting_clf.predict(X_test_scaled)
y_test_pred
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_test_pred.astype(np.int32)})
submission.to_csv(r'./submission.csv', index=False)
clfs = [log_clf, rnd_clf, svm_clf]
y_val_pred = np.empty((len(X_val), len(clfs)), dtype=np.float32)

for index, clf in enumerate(clfs):
    y_val_pred[:, index] = clf.predict(X_val)
    
y_val_pred
rnd_blender = RandomForestClassifier(n_estimators=200, oob_score=True)
rnd_blender.fit(y_val_pred, y_val)
rnd_blender.oob_score_
y_test_pred = np.empty((len(X_test_scaled), len(clfs)), dtype=np.float32)

for index, clf in enumerate(clfs):
    y_test_pred[:, index] = clf.predict(X_test_scaled)

y_test_pred
y_test_pred_blender = rnd_blender.predict(y_test_pred)
y_test_pred_blender
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_test_pred_blender.astype(np.int32)})
submission.to_csv(r'./submission_blender.csv', index=False)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.plot(X2D[:, 0][y==0], X2D[:, 1][y==0], 'b.')
plt.plot(X2D[:, 0][y==1], X2D[:, 1][y==1], 'r.')
plt.show()
from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components=3, kernel='rbf', gamma=0.06)
X_reduced = rbf_pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y)

plt.show()
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=3, n_neighbors=5)
X_reduced = lle.fit_transform(X)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y)

plt.show()
from sklearn.manifold import MDS
mds = MDS(n_components=3, random_state=42)
X_reduced_mds = mds.fit_transform(X)

from sklearn.manifold import Isomap
isomap = Isomap(n_components=3)
X_reduced_isomap = isomap.fit_transform(X)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, random_state=42)
X_reduced_tsne = tsne.fit_transform(X)
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_reduced_mds[:, 0], X_reduced_mds[:, 1], X_reduced_mds[:, 2], c=y)
plt.show()
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_reduced_isomap[:, 0], X_reduced_isomap[:, 1], X_reduced_isomap[:, 2], c=y)
plt.show()
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], X_reduced_tsne[:, 2], c=y)
plt.show()
