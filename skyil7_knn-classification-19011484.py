import seaborn as sns

import pandas as pd

import numpy as np

RANDOM_SEED = 777
data = pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv', index_col=0)

data.head()
sns.set()

sns.pairplot(data, hue='8', size=2.5)
data.describe()
x = data.drop('8', axis=1)

y = data['8']

print(x.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

print(x_train.shape)

print(x_test.shape)
# 데이터 표준화

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

x_train_std=scale.fit_transform(x_train)

x_test_std=scale.transform(x_test)
y_train.sum()
from imblearn.over_sampling import SMOTE, ADASYN

ada = ADASYN(random_state=RANDOM_SEED)

x_syn, y_syn = ada.fit_resample(x_train, y_train)



ada_std = ADASYN(random_state=RANDOM_SEED)

x_syn_std, y_syn_std = ada_std.fit_resample(x_train_std, y_train)



smt = SMOTE(random_state=RANDOM_SEED)

x_smt, y_smt = smt.fit_resample(x_train, y_train)



smt_std = SMOTE(random_state=RANDOM_SEED)

x_smt_std, y_smt_std = smt_std.fit_resample(x_train_std, y_train)
from sklearn.neighbors import KNeighborsClassifier

knn_syn = KNeighborsClassifier(n_neighbors=5, p=2)

knn_syn.fit(x_syn, y_syn)



knn_syn_std = KNeighborsClassifier(n_neighbors=5, p=2)

knn_syn_std.fit(x_syn_std, y_syn_std)



knn_smt = KNeighborsClassifier(n_neighbors=5, p=2)

knn_smt.fit(x_smt, y_smt)



knn_smt_std = KNeighborsClassifier(n_neighbors=5, p=2)

knn_smt_std.fit(x_smt_std, y_smt_std)



knn = KNeighborsClassifier(n_neighbors=5, p=2)

knn.fit(x_train, y_train)



knn_std = KNeighborsClassifier(n_neighbors=5, p=2)

knn_std.fit(x_train_std, y_train)

print('fit complete')
syn_pred = knn_syn.predict(x_test)

syn_std_pred = knn_syn_std.predict(x_test_std)

smt_pred = knn_smt.predict(x_test)

smt_std_pred = knn_smt_std.predict(x_test_std)

knn_pred = knn.predict(x_test)

knn_std_pred = knn_std.predict(x_test_std)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, syn_pred))

print(accuracy_score(y_test, syn_std_pred))

print(accuracy_score(y_test, smt_pred))

print(accuracy_score(y_test, smt_std_pred))

print(accuracy_score(y_test, knn_pred))

print(accuracy_score(y_test, knn_std_pred))
from sklearn.metrics import confusion_matrix#  오분류표 작성

conf1=confusion_matrix(y_true=y_test, y_pred=knn_std_pred)

conf2=confusion_matrix(y_true=y_test, y_pred=smt_pred)

conf3=confusion_matrix(y_true=y_test, y_pred=knn_pred)

print(conf1)

print(conf2)

print(conf3)
scaler = StandardScaler()

x_train_std = scale.fit_transform(x_train)

x_test_std = scale.transform(x_test)
knn = KNeighborsClassifier(n_neighbors=27, p=2)

knn.fit(x_train_std, y_train)



predict_knn = knn.predict(x_test_std)

print(accuracy_score(y_test, predict_knn))
test = pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv', index_col=0).drop('8', axis=1)

test.head()
test = scale.transform(test) # 표준화

pred = knn.predict(test)

pred[:5]
submit = pd.read_csv('../input/logistic-classification-diabetes-knn/submission_form.csv', index_col=0)

submit.head()
submit['Label'] = pred

submit['Label'] = submit['Label'].astype(int)

submit.head()
submit.to_csv('submit.csv')