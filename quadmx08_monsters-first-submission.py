import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split



%matplotlib inline
monster_train = pd.read_csv('../input/train.csv', index_col='id')

monster_test = pd.read_csv('../input/test.csv', index_col='id')
monster_train.head()
sns.pairplot(monster_train, size=1.5, hue='type')
msk = np.random.rand(len(monster_train)) < 0.8

monster_train_A = monster_train[msk]

monster_train_B = monster_train[~msk]



print('%d monsters in training subset A' % len(monster_train_A))

print('%d monsters in training subset B' % len(monster_train_B))
features = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']



def how_accurate(classifier, n=5):

    accuracy = []

    for i in range(n):

        classifier.fit(monster_train_A[features], monster_train_A['type'])

        preds = classifier.predict(monster_train_B[features])

        accuracy.append(np.where(preds==monster_train_B['type'], 1, 0).sum() / float(len(monster_train_B)))

    return np.mean(accuracy)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.neural_network import MLPClassifier
knn_clf = KNeighborsClassifier(n_neighbors=6)

tree_clf = DecisionTreeClassifier(max_depth=5)

forest_clf = RandomForestClassifier(max_depth=2, n_estimators=100, max_features=1)

gaussian_clf = GaussianProcessClassifier(4.0 * RBF(1.0), warm_start=True, n_jobs=-1)

neural_clf = MLPClassifier(alpha=1, max_iter=400)



clfs = {'K Nearest Neighbors': knn_clf,

        'Decision Tree': tree_clf,

        'Random Forest': forest_clf,

        'Gassian Process': gaussian_clf, 

        'Neural Network': neural_clf}
accuracy = {clf: how_accurate(clfs[clf], n=20) for clf in clfs}
for clf in clfs:

    print('%s: %s' % (clf.ljust(20), round(accuracy[clf],3)))
upper_quantile = 0.98

lower_quantile = .02
monster_train_A.columns.tolist()[:4]
for col in monster_train_A.columns.tolist()[:4]:

    upper_limit = monster_train_A[col].quantile(upper_quantile)

    lower_limit = monster_train_A[col].quantile(lower_quantile)

    

    monster_train_A.loc[monster_train_A[col]>upper_limit, col] = np.nan

    monster_train_A.loc[monster_train_A[col]<lower_limit, col] = np.nan

    

monster_train_A = monster_train_A.dropna()
accuracy_b = {clf: how_accurate(clfs[clf], n=20) for clf in clfs}
for clf in clfs:

    print('%s: %s' % (clf.ljust(20), round(accuracy_b[clf],3)))