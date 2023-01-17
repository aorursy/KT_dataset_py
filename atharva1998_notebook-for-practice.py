# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df.describe()
train_df.describe(include = 'object')
train_df.shape
print("Value Counts of Tickets: {}".format(train_df['Ticket'].value_counts()))
print("Value Counts of Cabin: {}".format(train_df['Cabin'].value_counts()))
import matplotlib.pyplot as plt

train_df['Ticket'].value_counts().plot(kind = 'bar')

plt.show()
train_df['Cabin'].value_counts().plot(kind = 'bar')

plt.show()
train_df['Fare'].plot(kind = 'bar')
train_df_1 = train_df.drop(["Name", "Ticket", "PassengerId", "Cabin"], axis = 1)
X_train = train_df_1.drop(["Survived"], axis = 1)

y_train = train_df_1[["Survived"]]
X_train["Age"] = X_train["Age"].fillna(X_train["Age"].mean())
X_train["Age"].describe()
X_train["Embarked"] = X_train["Embarked"].fillna(X_train["Embarked"].mode().iloc[0])
X_train["Embarked"].describe()
X_train.describe()
X_train.describe(include = 'object')
corr_mat = train_df_1.corr()

corr_mat
import seaborn as sns

sns.heatmap(corr_mat)
dum_df1 = pd.get_dummies(X_train["Embarked"])

X_train = X_train.drop(["Embarked"], axis = 1)

X_train = pd.concat([X_train, dum_df1], axis = 1)

dum_df2 = pd.get_dummies(X_train["Sex"])

X_train = X_train.drop(["Sex"], axis = 1)

X_train = pd.concat([X_train, dum_df2], axis = 1)

X_train.head()
dum_df3 = pd.get_dummies(X_train["Pclass"])

X_train = X_train.drop(["Pclass"], axis = 1)

X_train = pd.concat([X_train, dum_df3], axis = 1)

X_train.head()
X_train['Age'] = (X_train['Age'] - X_train['Age'].mean()) / X_train['Age'].std()

X_train['Fare'] = (X_train['Fare'] - X_train['Fare'].mean()) / X_train['Fare'].std()

X_train.head() 
from sklearn.decomposition import PCA

pca = PCA(0.9)

X_train_pca = pca.fit_transform(X_train)

no_of_components = pca.n_components_

var_ratio = pca.explained_variance_ratio_

plt.plot(var_ratio)

plt.xlabel('Features')

plt.ylabel('Proportion of variance explained by each feature.')

plt.title('The number of features are: {}'.format(no_of_components))

plt.show()
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

clf_l2 = LogisticRegression(penalty = 'l2', solver = 'lbfgs', random_state = 0)

clf_l2.fit(X_train, y_train)
scores_l2 = cross_val_score(clf_l2, X_train, y_train, cv = 3)

scores_l2
clf_l1 = LogisticRegression(penalty = 'l1', solver = 'liblinear', random_state = 0)

clf_l1.fit(X_train,y_train)

scores_l1 = cross_val_score(clf_l1, X_train, y_train, cv = 3)

scores_l1
print("L1 Regularisation Cross Validation Score is: " + str(scores_l1.mean()))

print("L2 Regularisation Cross Validation Score is: " + str(scores_l2.mean()))
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

parameters = {'n_neighbors':np.arange(1, 11)}

knn = KNeighborsClassifier()

clf_knn = GridSearchCV(knn, parameters, cv = 3)

clf_knn.fit(X_train, y_train)
clf_knn.cv_results_
knn_best_estimator = clf_knn.best_estimator_ 

knn_best_score = clf_knn.best_score_

print("The best estimator is: " + str(knn_best_estimator))

print("The best score is: " + str(knn_best_score))
from sklearn.svm import SVC

params_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

parameters = [{'C': params_range, 'kernel':['linear']}, {'C':params_range, 'kernel':['rbf'], 'gamma':params_range}]

svm = SVC(random_state = 0, probability = True)

clf_svm = GridSearchCV(estimator = svm, param_grid = parameters, cv = 3, n_jobs = -1, scoring = 'accuracy')

clf_svm.fit(X_train, y_train)
best_estimator_svm = clf_svm.best_estimator_

best_score_svm = clf_svm.best_score_

print(best_estimator_svm)

print(best_score_svm)
from sklearn import tree

tree = tree.DecisionTreeClassifier()

params_range = np.arange(1, 21)

parameters_tree = [{'criterion':['gini'], 'max_depth':params_range}, {'criterion':['entropy'], 'max_depth':params_range}]

clf_tree = GridSearchCV(estimator = tree, param_grid = parameters_tree, cv = 3, scoring = 'accuracy')

clf_tree.fit(X_train, y_train)

tree_best_estimator = clf_tree.best_estimator_

tree_best_score = clf_tree.best_score_

print(tree_best_estimator)

print(tree_best_score)
from sklearn.ensemble import RandomForestClassifier

params_range_n_estimators = np.arange(1, 21)

params_range_max_depth = np.arange(1, 21)

forest = RandomForestClassifier()

parameters_forest = [{'criterion':['gini'], 'n_estimators': params_range_n_estimators, 'max_depth':params_range_max_depth, 'oob_score':['True', 'False']}, {'criterion':['entropy'], 'n_estimators': params_range_n_estimators, 'max_depth':params_range_max_depth, 'oob_score':['True', 'False']}]

clf_forest = GridSearchCV(estimator = forest, param_grid = parameters_forest, cv = 3, scoring = 'accuracy')

clf_forest.fit(X_train, y_train)
clf_forest_best_estimator = clf_forest.best_estimator_

clf_forest_best_score = clf_forest.best_score_

print(clf_forest_best_estimator)

print(clf_forest_best_score)
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score

voting_clf = VotingClassifier(estimators = [('lr', clf_l2), ('svc', best_estimator_svm), ('knn', knn_best_estimator), ('rf', clf_forest_best_estimator)], voting = 'hard')

cross_validation_score = []

for clf in (clf_l2, clf_l1, best_estimator_svm, clf_forest_best_estimator, knn_best_estimator, voting_clf):

    clf.fit(X_train, y_train)

    cross_validation_score.append(clf)

    cross_validation_score.append(max(cross_val_score(clf, X_train, y_train, cv = 3)))

print(cross_validation_score)
test_df = pd.read_csv('../input/test.csv')
test_df.head()
test_df.describe(include = 'object')
test_df_1 = test_df.drop(["Name", "Ticket", "PassengerId", "Cabin"], axis = 1)
test_df_1.head()
test_df_1["Age"] = test_df_1["Age"].fillna(test_df_1["Age"].mean())
test_df_1["Fare"] = test_df_1["Fare"].fillna(test_df_1["Fare"].mean())
test_df_1.describe()
dum_df1_test = pd.get_dummies(test_df_1["Embarked"])

test_df_1 = test_df_1.drop(["Embarked"], axis = 1)

test_df_1 = pd.concat([test_df_1, dum_df1_test], axis = 1)

dum_df1_test = pd.get_dummies(test_df_1["Sex"])

test_df_1 = test_df_1.drop(["Sex"], axis = 1)

test_df_1 = pd.concat([test_df_1, dum_df1_test], axis = 1)
dum_df1_test = pd.get_dummies(test_df_1["Pclass"])

test_df_1 = test_df_1.drop(["Pclass"], axis = 1)

test_df_1 = pd.concat([test_df_1, dum_df1_test], axis = 1)
test_df_1.head()
test_df_1['Age'] = (test_df_1['Age'] - test_df_1['Age'].mean()) / test_df_1['Age'].std()

test_df_1['Fare'] = (test_df_1['Fare'] - test_df_1['Fare'].mean()) / test_df_1['Fare'].std()
test_df_1.head()
X_test_pca = pca.transform(test_df_1)
predictions = best_estimator_svm.predict(test_df_1)

submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':predictions})
submission.head()
filename = 'Titanic-Notebook-for-pratice-preds.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)