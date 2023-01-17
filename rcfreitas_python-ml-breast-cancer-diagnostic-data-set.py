import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/data.csv');



print("\n \t The data frame has {0[0]} rows and {0[1]} columns. \n".format(data.shape))

data.info()



data.head(3)
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)



data.info()
diagnosis_all = list(data.shape)[0]

diagnosis_categories = list(data['diagnosis'].value_counts())



print("\n \t The data has {} diagnosis, {} malignant and {} benign.".format(diagnosis_all, 

                                                                                 diagnosis_categories[0], 

                                                                                 diagnosis_categories[1]))
features_mean= list(data.columns[1:11])
plt.figure(figsize=(10,10))

sns.heatmap(data[features_mean].corr(), annot=True, square=True, cmap='coolwarm')

plt.show()
color_dic = {'M':'red', 'B':'blue'}

colors = data['diagnosis'].map(lambda x: color_dic.get(x))



sm = pd.scatter_matrix(data[features_mean], c=colors, alpha=0.4, figsize=((15,15)));



plt.show()
bins = 12

plt.figure(figsize=(15,15))

for i, feature in enumerate(features_mean):

    rows = int(len(features_mean)/2)

    

    plt.subplot(rows, 2, i+1)

    

    sns.distplot(data[data['diagnosis']=='M'][feature], bins=bins, color='red', label='M');

    sns.distplot(data[data['diagnosis']=='B'][feature], bins=bins, color='blue', label='B');

    

    plt.legend(loc='upper right')



plt.tight_layout()

plt.show()
plt.figure(figsize=(15,15))

for i, feature in enumerate(features_mean):

    rows = int(len(features_mean)/2)

    

    plt.subplot(rows, 2, i+1)

    

    sns.boxplot(x='diagnosis', y=feature, data=data, palette="Set1")



plt.tight_layout()

plt.show()
features_selection = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean']
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score



import time
diag_map = {'M':1, 'B':0}

data['diagnosis'] = data['diagnosis'].map(diag_map)
X = data.loc[:,features_mean]

y = data.loc[:, 'diagnosis']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



accuracy_all = []

cvs_all = []
from sklearn.linear_model import SGDClassifier



start = time.time()



clf = SGDClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))
from sklearn.svm import SVC, NuSVC, LinearSVC



start = time.time()



clf = SVC()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))



start = time.time()



clf = NuSVC()

clf.fit(X_train, y_train)

prediciton = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))



start = time.time()



clf = LinearSVC()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))
from sklearn.neighbors import KNeighborsClassifier



start = time.time()



clf = KNeighborsClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))
from sklearn.naive_bayes import GaussianNB



start = time.time()



clf = GaussianNB()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier



start = time.time()



clf = RandomForestClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))



start = time.time()



clf = ExtraTreesClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))



start = time.time()



clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("Dedicion Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))
X = data.loc[:,features_selection]

y = data.loc[:, 'diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



accuracy_selection = []

cvs_selection = []
from sklearn.linear_model import SGDClassifier



start = time.time()



clf = SGDClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_selection.append(accuracy_score(prediction, y_test))

cvs_selection.append(np.mean(scores))



print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))
from sklearn.svm import SVC, NuSVC, LinearSVC



start = time.time()



clf = SVC()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_selection.append(accuracy_score(prediction, y_test))

cvs_selection.append(np.mean(scores))



print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))



start = time.time()



clf = NuSVC()

clf.fit(X_train, y_train)

prediciton = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_selection.append(accuracy_score(prediction, y_test))

cvs_selection.append(np.mean(scores))



print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))



start = time.time()



clf = LinearSVC()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_selection.append(accuracy_score(prediction, y_test))

cvs_selection.append(np.mean(scores))



print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))
from sklearn.neighbors import KNeighborsClassifier



start = time.time()



clf = KNeighborsClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_selection.append(accuracy_score(prediction, y_test))

cvs_selection.append(np.mean(scores))



print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))
from sklearn.naive_bayes import GaussianNB



start = time.time()



clf = GaussianNB()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_selection.append(accuracy_score(prediction, y_test))

cvs_selection.append(np.mean(scores))



print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier



start = time.time()



clf = RandomForestClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_selection.append(accuracy_score(prediction, y_test))

cvs_selection.append(np.mean(scores))



print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))



start = time.time()



clf = ExtraTreesClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_selection.append(accuracy_score(prediction, y_test))

cvs_selection.append(np.mean(scores))



print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))



start = time.time()



clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_selection.append(accuracy_score(prediction, y_test))

cvs_selection.append(np.mean(scores))



print("Dedicion Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))
diff_accuracy = list(np.array(accuracy_selection) - np.array(accuracy_all))

diff_cvs = list(np.array(cvs_selection) - np.array(cvs_all))



d = {'accuracy_all':accuracy_all, 'accuracy_selection':accuracy_selection, 'diff_accuracy':diff_accuracy, 

     'cvs_all':cvs_all, 'cvs_selection':cvs_selection, 'diff_cvs':diff_cvs,}



index = ['SGD', 'SVC', 'NuSVC', 'LinearSVC', 'KNeighbors', 'GaussianNB', 'RandomForest', 'ExtraTrees', 'DecisionTree']



df = pd.DataFrame(d, index=index)
df
from sklearn.model_selection import GridSearchCV



X = data.loc[:,features_mean]

y = data.loc[:, 'diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



accuracy_all = []

csv_all = []
start = time.time()



parameters = {'priors':[[0.01, 0.99],[0.1, 0.9], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7],[0.35, 0.65], [0.4, 0.6]]}



clf = GridSearchCV(GaussianNB(), parameters, scoring = 'average_precision', n_jobs=-1)

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))



print("Best parameters: {0}".format(clf.best_params_))
start = time.time()



parameters = {'n_estimators':list(range(1,101)), 'criterion':['gini', 'entropy']}



clf = GridSearchCV(RandomForestClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))



print("Best parameters: {0} \n".format(clf.best_params_))



start = time.time()



clf = GridSearchCV(ExtraTreesClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))



print("Best parameters: {0} \n".format(clf.best_params_))



start = time.time()



parameters = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random']}



clf = GridSearchCV(DecisionTreeClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)



end = time.time()



accuracy_all.append(accuracy_score(prediction, y_test))

cvs_all.append(np.mean(scores))



print("Dedicion Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))



print("Best parameters: {0} \n".format(clf.best_params_))