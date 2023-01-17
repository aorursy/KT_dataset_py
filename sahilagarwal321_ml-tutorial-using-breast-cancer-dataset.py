import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import metrics, preprocessing

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
data = pd.read_csv("../input/data.csv")
data.sample(10)
del data['id']

del data['Unnamed: 32']
data.sample(10)
data.info()
data['diagnosis'].unique()
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data.sample(10)
sns.countplot(data.diagnosis, label='Count')
print(list(data.columns))
features_mean = list(data.columns[1:11])

features_se = list(data.columns[11:21])

features_worst =list(data.columns[21:31])

print("---------------- features_mean -------------------------------------------------------")

print(features_mean)

print("\n---------------- features_se (Standard Error) -------------------------------------------------------")

print(features_se)

print("\n---------------- features_worst --------------------------------------------------------")

print(features_worst)
corr = data[features_mean].corr().abs()

lower_right_ones = np.tril(np.ones(corr.shape, dtype='bool'), k=-1)

correlations = corr.where(lower_right_ones)

correlations
plt.figure(figsize=(12,12))

sns.heatmap(correlations, annot=True, cmap='RdBu_r', fmt= '.2f', vmax=1, vmin=-1)

plt.xticks(rotation=60)
THRESHOLD_VALUE = 0.85

list(i for i in (correlations[correlations.gt(THRESHOLD_VALUE)].stack().index) if i[0] is not i[1])
correlations[correlations.gt(THRESHOLD_VALUE)].stack().sort_values(ascending = False)
corr = data[features_se].corr().abs()

lower_right_ones = np.tril(np.ones(corr.shape, dtype='bool'), k=-1)

correlations = corr.where(lower_right_ones)

plt.figure(figsize=(12,12))

sns.heatmap(correlations, annot=True, cmap='RdBu_r', fmt= '.2f', vmax=1, vmin=-1)

plt.xticks(rotation=60)
THRESHOLD_VALUE = 0.85

correlations[correlations.gt(THRESHOLD_VALUE)].stack().sort_values(ascending = False)
corr = data[features_worst].corr().abs()

lower_right_ones = np.tril(np.ones(corr.shape, dtype='bool'), k=-1)

correlations = corr.where(lower_right_ones)

plt.figure(figsize=(12,12))

sns.heatmap(correlations, annot=True, cmap='RdBu_r', fmt= '.2f', vmax=1, vmin=-1)

plt.xticks(rotation=60)
THRESHOLD_VALUE = 0.85

correlations[correlations.gt(THRESHOLD_VALUE)].stack().sort_values(ascending = False)
to_remove = [

    'concave points_meanr', 'compactness_mean', 'perimeter_mea', 'area_mean',

    'concave points_worst', 'compactness_worst', 'perimeter_worst', 'area_worst',

    'perimeter_se', 'area_se'

]

to_use = [e for e in data.columns if e not in to_remove]

print(to_use)
reduced_data = data[to_use]

reduced_data.sample(10)
X = reduced_data.loc[:, 'radius_mean': 'fractal_dimension_worst']

Y = reduced_data['diagnosis']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
sc = preprocessing.StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
pca = PCA(.95)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
svc = SVC()

gaussian_nb = GaussianNB()

decision_tree_classifier = DecisionTreeClassifier()

random_forest_classifier = RandomForestClassifier()

logistic_regression = LogisticRegression()

k_neighbors_classifier = KNeighborsClassifier()
svc.fit(X_train,Y_train)

gaussian_nb.fit(X_train,Y_train)

decision_tree_classifier.fit(X_train,Y_train)

random_forest_classifier.fit(X_train,Y_train)

logistic_regression.fit(X_train,Y_train)

k_neighbors_classifier.fit(X_train,Y_train)
print("svc - {0:.3f}".format(svc.score(X_test, Y_test)))

print("gaussian_nb - {0:.3f}".format(gaussian_nb.score(X_test, Y_test)))

print("decision_tree_classifier - {0:.3f}".format(decision_tree_classifier.score(X_test, Y_test)))

print("random_forest_classifier - {0:.3f}".format(random_forest_classifier.score(X_test, Y_test)))

print("logistic_regression - {0:.3f}".format(logistic_regression.score(X_test, Y_test)))

print("k_neighbors_classifier - {0:.3f}".format(k_neighbors_classifier.score(X_test, Y_test)))
