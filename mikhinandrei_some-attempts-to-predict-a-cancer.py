import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd
data = pd.read_csv('../input/data.csv').drop(['id', 'Unnamed: 32'], axis=1)

data.info()
data.describe()
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

corr_matrix = data.corr()

sns.heatmap(corr_matrix, cmap='Blues')
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(60, 30))



for idx, feat in enumerate(data.columns):

    sns.boxplot(x='diagnosis', y=feat, data=data, ax=axes[idx // 8, idx % 8], hue='diagnosis')

    axes[idx // 8, idx % 8].legend()

    axes[idx // 8, idx % 8].set_xlabel('diagnosis')

    axes[idx // 8, idx % 8].set_ylabel(feat)
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler



scs = StandardScaler()

data_scaled = scs.fit_transform(data)

t = TSNE(random_state=20)

tsne_repr = t.fit_transform(data_scaled)

plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=data['diagnosis'].map({0: 'blue', 1: 'orange'}))
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(data.drop('diagnosis', axis=1), data['diagnosis'])



dtc = DecisionTreeClassifier(random_state=17, max_depth=4)

dtc.fit(X_train, y_train)

pred = dtc.predict(X_test)

print('Precision of decision tree is {}%'.format(accuracy_score(y_test, pred) * 100))



knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print('Precision of KNN tree is {}%'.format(accuracy_score(y_test, pred) * 100))