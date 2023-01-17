import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
labels = pd.read_csv('../input/actual.csv', index_col = 'patient')

test = pd.read_csv('../input/data_set_ALL_AML_independent.csv')

train = pd.read_csv('../input/data_set_ALL_AML_train.csv')

train.head()
cols = [col for col in test.columns if 'call' in col]

test = test.drop(cols, 1)



cols = [col for col in train.columns if 'call' in col]

train = train.drop(cols, 1)



train.head()
train = train.T

test = test.T

train.head()
train = train.drop(['Gene Description', 'Gene Accession Number'])

test = test.drop(['Gene Description', 'Gene Accession Number'])



train.head()
labels = labels.replace({'ALL':0,'AML':1})

labels_train = labels[labels.index <= 38]

labels_test = labels[labels.index > 38]



train = train.replace(np.inf, np.nan)

train = train.fillna(value = train.values.mean())



test = test.replace(np.inf, np.nan)

test = test.fillna(value = test.values.mean())
df_all = train.append(test, ignore_index=True)
from sklearn import preprocessing

X_all = preprocessing.StandardScaler().fit_transform(df_all)
from sklearn.decomposition import PCA

pca = PCA(n_components=50, random_state=42)

X_pca = pca.fit_transform(X_all)

print(X_pca.shape)
cum_sum = pca.explained_variance_ratio_.cumsum()

cum_sum = cum_sum*100



plt.bar(range(50), cum_sum)

plt.title("Around 90% of variance is explained by the First 50 columns ");
from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier as XGB

from sklearn.metrics import accuracy_score, confusion_matrix
model = SVC()

model.fit(X_pca[:38,:], labels_train.values.ravel())
pred = model.predict(X_pca[38:,:])

print('Accuracy: ', accuracy_score(labels_test, pred))
confusion_matrix(labels_test, pred)
xgb = XGB(max_depth=10, loss='exponential', n_estimators=100, learning_rate=0.02, random_state=42)

xgb.fit(X_pca[:38,:], labels_train.values.ravel())
pred = xgb.predict(X_pca[38:,:])

print('Accuracy: ', accuracy_score(labels_test, pred))
confusion_matrix(labels_test, pred)