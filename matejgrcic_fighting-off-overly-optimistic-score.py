import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
sns.set(style='whitegrid')
data = pd.read_csv('../input/heart.csv')

print(F"Null values? {data.isnull().values.any()}")
data.head()
plt.figure(figsize=(7, 5))

count_per_class = [len(data[data['target'] == 0]),len(data[data['target'] == 1])]

labels = [0, 1]

colors = ['yellowgreen', 'lightblue']

explode = (0.05, 0.1)

plt.pie(count_per_class, explode=explode, labels=labels, 

        colors=colors,autopct='%4.2f%%',shadow=True, startangle=45)

plt.title('Examples per class')

plt.axis('equal')

plt.show()
plt.figure(figsize=(7, 5))

count_per_class = [len(data[data['sex'] == 0]),len(data[data['sex'] == 1])]

labels = ['Female', 'Male']

colors = ['lightgreen', 'gold']

explode = (0.05, 0.1)

plt.pie(count_per_class, explode=explode, labels=labels, 

        colors=colors,autopct='%4.2f%%',shadow=True, startangle=70)

plt.title('Gender shares')

plt.axis('equal')

plt.show()
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

sns.kdeplot(data['age'], data['sex'], shade=True)

plt.title('Age-sex density estimate')

plt.subplot(1, 2, 2)

sns.distplot(data['age'])

plt.title('Age distribution')

plt.show()
plt.figure(figsize=(8, 6))

sns.distplot(data[data.target == 0]['chol'], label='without heart disease')

sns.distplot(data[data.target == 1]['chol'], label='with heart disease')

plt.xlabel('serum cholestoral in mg/dl')

plt.title('serum cholestoral per class')

plt.legend()

plt.show()
plt.figure(figsize=(8, 6))

sns.distplot(data[data.target == 0]['thalach'], label='without heart disease')

sns.distplot(data[data.target == 1]['thalach'], label='with heart disease')

plt.title('maximum heart rate achieved per class')

plt.xlabel('maximum heart rate achieved')

plt.legend()

plt.show()
plt.figure(figsize=(12,8))

sns.heatmap(data.corr(), annot=True, linewidths=2, cmap="YlGnBu")

plt.show()
data.groupby('target')['trestbps'].describe()
ax2 = sns.jointplot("target", "trestbps", data=data, kind="reg", color='r')

ax2.set_axis_labels('target','resting blood pressure')

plt.show()
X = data.values[:, :13]

y = data.values[:, 13]
import eli5

from sklearn.linear_model import LogisticRegression

from eli5.sklearn import PermutationImportance



logistic_regression = LogisticRegression(penalty='l1')

logistic_regression.fit(X, y)

perm_imp = PermutationImportance(logistic_regression, random_state=42).fit(X, y)

eli5.show_weights(perm_imp, feature_names = data.columns.tolist()[:13])

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.metrics import accuracy_score



def nested_kfold_cv(model, param_grid, X, y, outer_metric=accuracy_score,

                    scoring='accuracy' , k1=10, k2=3, verbose = 1, n_jobs=3, shuffle=True):

    scores = []

    estimators = []

    kf = KFold(n_splits=k1, shuffle=shuffle)

    for train_index, test_index in kf.split(X):

        X_train = X[train_index]

        X_test = X[test_index]

        y_train = y[train_index]

        y_test = y[test_index]

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=k2,

                                   verbose=verbose, n_jobs=n_jobs, scoring=scoring)

        grid_search.fit(X=X_train, y=y_train)

        estimator = grid_search.best_estimator_

        estimators.append(estimator)

        estimator.fit(X_train, y_train)

        scores.append(outer_metric(estimator.predict(X_test), y_test))

    return estimators, scores
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import recall_score, confusion_matrix
tree_model = AdaBoostClassifier(

    base_estimator=DecisionTreeClassifier(max_depth=1),

    random_state=42

)
tree_params = {

    'n_estimators': [25, 50, 75]

}

estimators, tree_scores = nested_kfold_cv(tree_model, tree_params, X, y, outer_metric=recall_score,

                                     scoring='f1' , k1=10, k2=5, verbose = 0, n_jobs=4, shuffle=True)
print(f"Average recall: {np.mean(tree_scores)}")
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.svm import SVC
svm_model = Pipeline(steps=[

    ('standard_scaler', StandardScaler()),

    ('feature_selection', SelectKBest(f_classif)), # params: k

    ('svm', SVC(kernel='rbf', random_state=42)) # params: gamma, C

])
svm_grid = {

    'feature_selection__k': [10, 12, 13],

    'svm__C': [3, 5, 10, 15, 20, 25, 30, 35],

    'svm__gamma': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],

    

}

estimators, svm_scores = nested_kfold_cv(svm_model, svm_grid, X, y, outer_metric=recall_score,

                                     scoring='f1' , k1=10, k2=5, verbose = 0, n_jobs=4, shuffle=True)
print(f"Average recall: {np.mean(svm_scores)}")
from sklearn.linear_model import LogisticRegression
log_model = Pipeline(steps=[

    ('feature_selection', SelectKBest(f_classif)), # params: k

    ('log', LogisticRegression()) # params:  C

])
log_grid = {

    'log__C': [0.01, 0.1, 0.5, 1, 3, 5],

    'feature_selection__k': [5, 9, 10, 12, 13],

}

estimators, lr_scores = nested_kfold_cv(log_model, log_grid, X, y, outer_metric=recall_score,

                                     scoring='f1' , k1=10, k2=5, verbose = 0, n_jobs=4, shuffle=True)
print(f"Average recall: {np.mean(lr_scores)}")
from sklearn.neighbors import KNeighborsClassifier
knn_model = Pipeline(steps=[

    ('standard_scaler', StandardScaler()),

    ('knn', KNeighborsClassifier(weights='distance')) # params: n_neighbors

])
knn_grid = {

    'knn__n_neighbors': [3, 5, 7, 10, 12, 15, 17, 20],

}

estimators, knn_scores = nested_kfold_cv(knn_model, knn_grid, X, y, outer_metric=recall_score,

                                     scoring='f1' , k1=10, k2=5, verbose = 0, n_jobs=4, shuffle=True)
print(f"Average recall: {np.mean(knn_scores)}")
from sklearn.neural_network import MLPClassifier
nn_model = Pipeline(steps=[

    ('standard_scaler', StandardScaler()),

    ('nn', MLPClassifier(max_iter=400)) # params:

])
nn_grid = {

    'nn__solver': ['adam', 'lbfgs']

}

estimators, nn_scores = nested_kfold_cv(nn_model, nn_grid, X, y, outer_metric=recall_score,

                                     scoring='f1' , k1=10, k2=5, verbose = 0, n_jobs=4, shuffle=True)
print(f"Average recall: {np.mean(nn_scores)}")
results = pd.DataFrame({'KNN': knn_scores, 'Logistic regression': lr_scores, 'SVC': svm_scores, 'AdaBoost': tree_scores, 'Neural network': nn_scores})

results.boxplot(figsize=(8, 6))