import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
filename = '../input/train.csv'
df = pd.read_csv(filename, sep=',', index_col=0)
df.head()
df.corr()
gender = df.Sex.map({'male':1, 'female':0})
df['Sex'] = gender
df.head()
df.info()
df.drop('Cabin', axis=1, inplace=True)
df.info()
df.Embarked.value_counts()
df.Embarked.fillna('O', inplace=True) 
df['Embarked'] = df.Embarked.map({'S':0, 'C':1, 'Q':2, 'O':3})
df.Age.fillna(df.Age.mean(), inplace=True) 
df.drop('Ticket', axis=1, inplace=True) 
df.drop('Name', axis=1, inplace=True) 
df.info()
y = df.Survived.values
df.drop('Survived', axis=1, inplace=True)
y = list(y)
X = df.values
len(y)
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_pred, y_test))
confusion_matrix(y_pred, y_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)
print("Baris x Kolom matriks X baru :", X_reduced.shape)


import pylab as plt
%matplotlib inline
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
           cmap='RdYlBu')

print("Arti dari 2 componen:")
for component in pca.components_:
    print(" + ".join("%.3f x %s" % (value, name)
                     for value, name in zip(component,
                                            df.columns)))
