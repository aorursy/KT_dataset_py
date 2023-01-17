# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()
df.type.unique()
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

from sklearn.manifold import TSNE

from sklearn.svm import SVC

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score

%matplotlib inline
col_n1 = df.columns[1:5]

y_n = ['type']

print(col_n1)
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df.fillna(-1, inplace=True)

df_test.fillna(-1, inplace=True)
sns.pairplot(df, hue='type')
lab = LabelEncoder()

df['col_lab'] = lab.fit_transform(df['color'])

ff = PolynomialFeatures(degree=2)

a = ff.fit_transform(df[col_n1])

df_t = pd.DataFrame(a)

df = pd.concat([df,df_t],axis=1)

df.head()
col_n=[]

for a in df.columns:

    if a not in ['id', 'color', 'type']:

        col_n.append(a)
print(col_n)

ts = TSNE()

m = ts.fit_transform(df[col_n])

df_t = pd.DataFrame(m)

df_t.columns = ['x', 'y']

df = pd.concat([df,df_t],axis=1)
df['c'] = 'green'

df.ix[df['type'] == 'Ghoul', 'c'] = 'red'

df.ix[df['type'] == 'Ghost', 'c'] = 'blue'

plt.scatter(df['x'], df['y'], c = df['c'])
X_tr, X_te, y_tr, y_te = train_test_split(df[col_n], df[y_n])
clf = RandomForestClassifier(n_estimators=150)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)
accuracy_score(y_pred=pred, y_true=y_te)
svm = SVC(C=20, decision_function_shape='ovr')


svm.fit(X_tr, y_tr)
pred2 = svm.predict(X_te)
accuracy_score(y_pred=pred2, y_true=y_te)
labels = df['type']

scores_f = cross_val_score(clf, df[col_n], labels, cv=10, n_jobs=-1)

scores_s = cross_val_score(svm, df[col_n], labels, cv=10, n_jobs=-1)

print(scores_f.mean())

print(scores_f)

print(scores_s.mean())

print(scores_s)
ff = PolynomialFeatures()

a = ff.fit_transform(df[col_n1])

df_t = pd.DataFrame(a)

df_test = pd.concat([df_test,df_t],axis=1)

print(col_n)

print(df_test.head())
df_test['type'] = clf.predict(df_test[col_n])
df_test.dtypes
df.dtypes
df_test.fillna(-1, inplace=True)
np.any(np.isnan(df_test[col_n]))
df_test[['id', 'type']].to_csv('test.csv')
print(check_output(["ls", "../input"]).decode("utf8"))