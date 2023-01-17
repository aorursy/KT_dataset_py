data_url = 'https://raw.githubusercontent.com/pkmklong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/master/data.csv'
import numpy as np

import pandas as pd

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt
df = pd.read_csv(data_url)

df.head()
df.columns
sns.countplot(df['diagnosis'])

plt.show()
df.drop(['Unnamed: 32'], axis = 1, inplace = True)
df.head()
df.drop(['id'], axis = 1, inplace = True)
df.head()
df.columns
X = df.iloc[:, 1:].values

y = df['diagnosis'].values
X.shape
y.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 0)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.decomposition import PCA



pca = PCA(n_components = 1)



X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(max_depth = 2, random_state = 0)

clf.fit(X_train, y_train)



# Predicting the Test set results

y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



cm = confusion_matrix(y_test, y_pred)

print(cm)

print('Accuracy -> '+ str(accuracy_score(y_test, y_pred)))
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 0)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2, random_state = 0)
tsne_obj = tsne.fit_transform(X_train)
tsne_df = pd.DataFrame({'X' : tsne_obj[:,0],

                       'Y' : tsne_obj[:,1],

                        'classification' : y_train

                       })
tsne_df.head()
tsne_df['classification'].value_counts()
plt.figure(figsize = (10,10))

sns.scatterplot(x = 'X', y = 'Y', data = tsne_df)

plt.show()
plt.figure(figsize = (10,10))

sns.scatterplot(x = "X", y = 'Y', hue = 'classification', legend = 'full', data = tsne_df)

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 0)



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



lda = LDA(n_components = 1)

X_train = lda.fit_transform(X_train, y_train)

X_test = lda.transform(X_test)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(max_depth = 2, random_state = 0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



cm = confusion_matrix(y_test, y_pred)

print(cm)



print('Accuracy -> ' + str(accuracy_score(y_test, y_pred)))