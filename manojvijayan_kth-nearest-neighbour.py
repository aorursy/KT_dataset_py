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



m_df = pd.read_csv('../input/mushrooms.csv',low_memory=False)
m_df.head(5)
m_df.info()
m_df.describe()
a = m_df.columns.values
a
m_df[a[:11]].describe()
m_df[a[11:]].describe()
m_df.drop('veil-type', axis=1,inplace=True)
b = m_df.columns.values
lst = ['cap-shape', 'cap-surface', 'cap-color', 'odor',  'gill-color', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
lst
m_df[b[:11]].describe()
m_df[b[11:]].describe()
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
m_df['class'] = lb.fit_transform(m_df['class'])
m_df['bruises'] = lb.fit_transform(m_df['bruises'])
m_df['gill-attachment'] = lb.fit_transform(m_df['gill-attachment'])
m_df['gill-spacing'] = lb.fit_transform(m_df['gill-spacing'])
m_df['gill-size'] = lb.fit_transform(m_df['gill-size'])
m_df['stalk-shape'] = lb.fit_transform(m_df['stalk-shape'])
m_df[b[:11]].head()
for cols in lst:

        m_df = pd.concat([m_df, pd.get_dummies(m_df[cols],drop_first=True,prefix=cols, prefix_sep='_')], axis=1)

        m_df.drop(cols, inplace=True, axis=1)

m_df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(m_df.iloc[:,1:].as_matrix(), m_df.iloc[:,0].values, test_size=0.30, random_state=101)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=1, weights='uniform' )
kn.fit(X_train,y_train)
pred = kn.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

pca.fit(X_train)
pca.explained_variance_
X_train_PCA = pca.transform(X_train)

X_test_PCA = pca.transform(X_test)
kn2 = KNeighborsClassifier(n_neighbors=1, weights='uniform' )
kn2.fit(X_train_PCA, y_train)
pred2 = kn2.predict(X_test_PCA)
print(confusion_matrix(y_test, pred2))

print(classification_report(y_test, pred2))
import matplotlib.pyplot as plt
for i in range(0, X_test_PCA.shape[0]):

    if y_test[i] == 0:

        c1 = plt.scatter(X_test_PCA[i,0],X_test_PCA[i,1],c='g',marker='+')

    elif y_test[i] == 1:

        c2 = plt.scatter(X_test_PCA[i,0],X_test_PCA[i,1],c='r',marker='o')

plt.legend([c1, c2], ['E', 'P'])

plt.title('PCA vs Classification Actual')