from sklearn.datasets import load_wine

import pandas as pd

import numpy as np

np.set_printoptions(precision=4)

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)

y = pd.Categorical.from_codes(wine.target, wine.target_names)
X.shape
X.head()
wine.target_names
df = X.join(pd.Series(y, name='class'))
class_feature_means = pd.DataFrame(columns=wine.target_names)

for c, rows in df.groupby('class'):

    class_feature_means[c] = rows.mean()

class_feature_means
within_class_scatter_matrix = np.zeros((13,13))

for c, rows in df.groupby('class'):

    rows = rows.drop(['class'], axis=1)

    

    s = np.zeros((13,13))
for index, row in rows.iterrows():

        x, mc = row.values.reshape(13,1), class_feature_means[c].values.reshape(13,1)

        

        s += (x - mc).dot((x - mc).T)    

within_class_scatter_matrix += s
feature_means = df.mean()

between_class_scatter_matrix = np.zeros((13,13))
for c in class_feature_means:    

    n = len(df.loc[df['class'] == c].index)

    

    mc, m = class_feature_means[c].values.reshape(13,1), feature_means.values.reshape(13,1)

    

    between_class_scatter_matrix += n * (mc - m).dot((mc - m).T)

    eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))

pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]

pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

for pair in pairs:

    print(pair[0])
eigen_value_sums = sum(eigen_values)

print('Explained Variance')

for i, pair in enumerate(pairs):

    print('Eigenvector {}: {}'.format(i, (pair[0]/eigen_value_sums).real))
w_matrix = np.hstack((pairs[0][1].reshape(13,1), pairs[1][1].reshape(13,1))).real
X_lda = np.array(X.dot(w_matrix))

le = LabelEncoder()

y = le.fit_transform(df['class'])
plt.xlabel('LD1')

plt.ylabel('LD2')

plt.scatter(

    X_lda[:,0],

    X_lda[:,1],

    c=y,

    cmap='rainbow',

    alpha=0.7,

    edgecolors='b'

)
w_matrix = np.hstack((pairs[0][1].reshape(13,1), pairs[1][1].reshape(13,1))).real
le = LabelEncoder()

y = le.fit_transform(df['class'])
plt.xlabel('LD1')

plt.ylabel('LD2')

plt.scatter(

    X_lda[:,0],

    X_lda[:,1],

    c=y,

    cmap='rainbow',

    alpha=0.7,

    edgecolors='b'

)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

X_lda = lda.fit_transform(X, y)
lda.explained_variance_ratio_
plt.xlabel('LD1')

plt.ylabel('LD2')

plt.scatter(

    X_lda[:,0],

    X_lda[:,1],

    c=y,

    cmap='rainbow',

    alpha=0.7,

    edgecolors='b'

)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X, y)
pca.explained_variance_ratio_
plt.xlabel('PC1')

plt.ylabel('PC2')

plt.scatter(

    X_pca[:,0],

    X_pca[:,1],

    c=y,

    cmap='rainbow',

    alpha=0.7,

    edgecolors='b'

)
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, random_state=1)
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

confusion_matrix(y_test, y_pred)