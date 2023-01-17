import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier

from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

%matplotlib inline
data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
data.head()
X = data.iloc[:,:-2] # features
y = data.iloc[:,-1]  # class

X_test = test_data.iloc[:,:-2] # features
y_test = test_data.iloc[:,-1]  # class
X.info()
feature_names = X.columns
data.Activity.unique()
class_names = ['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
y.value_counts()
plt.figure(figsize=(14,5))
ax = sns.countplot(y, label = "Count", palette = "Set3")
LAYING, STANDING, SITTING, WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS = y.value_counts()
dummy_classifier = DummyClassifier(strategy="most_frequent")
y_pred = dummy_classifier.fit(X, y).predict(X_test)
dummy_classifier.score(X_test, y_test)
cm = confusion_matrix(y_test, dummy_classifier.predict(X_test), labels = class_names)
sns.heatmap(cm, annot=True, fmt="d", xticklabels = class_names, yticklabels = class_names)
rf_for_refcv = RandomForestClassifier() 
rfecv0 = RFECV(estimator = rf_for_refcv, step = 1, cv = 5, scoring = 'accuracy')   #5-fold cross-validation
rfecv0 = rfecv0.fit(X, y)

print('Optimal number of features :', rfecv0.n_features_)
print('Best features :', X.columns[rfecv0.support_])
rfecv_X = X[X.columns[rfecv0.support_]]  # rfecv_X.shape = (7352, 550)
rfecv_X_test = X_test[X.columns[rfecv0.support_]]  # rfecv_X_test.shape = (2947, 550)
rf_for_emb = RandomForestClassifier()      
rf_for_emb = rf_for_emb.fit(X, y)
importances = rf_for_emb.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_for_emb.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
indices
h_indices = indices[:100]  # 100 best features
th_indices = indices[:300]  # 300 best features
plt.figure(1, figsize=(25, 13))
plt.title("Feature importances")
plt.bar(range(100), importances[h_indices], color="g", yerr=std[h_indices], align="center")
plt.xticks(range(100), X.columns[h_indices],rotation=90)
plt.xlim([-1, 100])
plt.show()
hundred_X = X[X.columns[h_indices]]
hundred_X_test = X_test[X.columns[h_indices]]

threeh_X = X[X.columns[th_indices]]
threeh_X_test = X_test[X.columns[th_indices]]
decision_tree00 = tree.DecisionTreeClassifier()
dtclf00 = decision_tree00.fit(X, y)
dtclf00.score(X_test, y_test)
decision_tree01 = tree.DecisionTreeClassifier()
dtclf01 = decision_tree01.fit(hundred_X, y)
dtclf01.score(hundred_X_test, y_test)
decision_tree02 = tree.DecisionTreeClassifier()
dtclf02 = decision_tree02.fit(threeh_X, y)
dtclf02.score(threeh_X_test, y_test)
decision_tree03 = tree.DecisionTreeClassifier()
dtclf03 = decision_tree03.fit(rfecv_X, y)
dtclf03.score(rfecv_X_test, y_test)
decision_tree04 = tree.DecisionTreeClassifier(min_samples_leaf=4)
dtclf04 = decision_tree04.fit(X, y)
dtclf04.score(X_test, y_test)
decision_tree05 = tree.DecisionTreeClassifier(min_samples_leaf=6)
dtclf05 = decision_tree05.fit(X, y)
dtclf05.score(X_test, y_test)
decision_tree06 = tree.DecisionTreeClassifier(min_samples_leaf=6)
dtclf06 = decision_tree06.fit(rfecv_X, y)
dtclf06.score(rfecv_X_test, y_test)
decision_tree07 = tree.DecisionTreeClassifier(min_samples_leaf=6)
dtclf07 = decision_tree07.fit(threeh_X, y)
dtclf07.score(threeh_X_test, y_test)
k_range = range(3, 50)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knnclf = knn.fit(X, y)
    scores.append(knnclf.score(X_test, y_test))

plt.figure(figsize=(14,5))
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
knn0 = KNeighborsClassifier(n_neighbors = 18)
knnclf0 = knn0.fit(X, y)
knnclf0.score(X_test, y_test)
knn1 = KNeighborsClassifier(n_neighbors = 19)
knnclf1 = knn1.fit(threeh_X, y)
knnclf1.score(threeh_X_test, y_test)
knn2 = KNeighborsClassifier(n_neighbors = 28)
knnclf2 = knn2.fit(rfecv_X, y)
knnclf2.score(rfecv_X_test, y_test)
mlp0 = MLPClassifier(hidden_layer_sizes=(15, 15))
mlp0 = mlp0.fit(X, y)
mlp0.score(X_test, y_test)
mlp1 = MLPClassifier(hidden_layer_sizes=(15, 15))
mlp1 = mlp1.fit(rfecv_X, y)
mlp1.score(rfecv_X_test, y_test)
mlp2 = MLPClassifier(hidden_layer_sizes=(15, 15))
mlp2 = mlp2.fit(rfecv_X, y)
mlp2.score(rfecv_X_test, y_test)
mlp3 = MLPClassifier(hidden_layer_sizes=(15, 15, 15))
mlp3 = mlp3.fit(X, y)
mlp3.score(X_test, y_test)
mlp4 = MLPClassifier(hidden_layer_sizes=(20, 20))
mlp4 = mlp4.fit(X, y)
mlp4.score(X_test, y_test)
mlp5 = MLPClassifier(hidden_layer_sizes=(30, 30))
mlp5 = mlp5.fit(X, y)
mlp5.score(X_test, y_test)
svcclf0 = SVC()
svcclf0 = svcclf0.fit(X, y)
svcclf0.score(X_test, y_test)
svcclf1 = SVC()
svcclf1 = svcclf1.fit(hundred_X, y)
svcclf1.score(hundred_X_test, y_test)
svcclf2 = SVC()
svcclf2 = svcclf2.fit(threeh_X, y)
svcclf2.score(threeh_X_test, y_test)
svcclf3 = SVC()
svcclf3 = svcclf3.fit(rfecv_X, y)
svcclf3.score(rfecv_X_test, y_test)
RFclf0 = RandomForestClassifier()
RFclf0 = RFclf0.fit(X, y)
RFclf0.score(X_test, y_test)
RFclf1 = RandomForestClassifier()
RFclf1 = RFclf1.fit(rfecv_X, y)
RFclf1.score(rfecv_X_test, y_test)
x_range = range(10, 25)
scores2 = []
for x in x_range:
    RFclf2 = RandomForestClassifier(n_estimators = x)
    RFclf2 = RFclf2.fit(X, y)
    scores2.append(RFclf2.score(X_test, y_test))

plt.plot(x_range, scores2)
plt.xlabel('Number of trees in the forest')
plt.ylabel('Testing Accuracy')
RFclf3 = RandomForestClassifier(n_estimators = 20)
RFclf3 = RFclf3.fit(X, y)
RFclf3.score(X_test, y_test)
RFclf4 = RandomForestClassifier(n_estimators = 20)
RFclf4 = RFclf4.fit(rfecv_X, y)
RFclf4.score(rfecv_X_test, y_test)
RFclf5 = RandomForestClassifier(n_estimators = 16)
RFclf5 = RFclf5.fit(threeh_X, y)
RFclf5.score(threeh_X_test, y_test)
bag0 = BaggingClassifier()
bag0 = bag0.fit(X, y)
bag0.score(X_test, y_test)
bag1 = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3))
bag1 = bag1.fit(X, y)
bag1.score(X_test, y_test)
bag2 = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=28))
bag2 = bag2.fit(X, y)
bag2.score(X_test, y_test)
ABclf0 = AdaBoostClassifier() #n_estimators=50, learning_rate = 1
ABclf0 = ABclf0.fit(X, y)
ABclf0.score(X_test, y_test)
ABclf1 = AdaBoostClassifier(n_estimators=200, learning_rate = 0.5)
ABclf1 = ABclf1.fit(X, y)
ABclf1.score(X_test, y_test)
ABclf2 = AdaBoostClassifier(n_estimators=400, learning_rate = 0.5)
ABclf2 = ABclf2.fit(X, y)
ABclf2.score(X_test, y_test)
voteclf1 = VotingClassifier(
    estimators=[('knn0', KNeighborsClassifier(n_neighbors = 19)), 
                ('mlp0', MLPClassifier(hidden_layer_sizes=(15, 15))),
                ('mlp4', MLPClassifier(hidden_layer_sizes=(20, 20))),
                ('RFclf1', RandomForestClassifier())], 
    voting='soft')
voteclf1 = voteclf1.fit(X, y)
voteclf1.score(X_test, y_test)
voteclf2 = VotingClassifier(
    estimators=[('knn0', KNeighborsClassifier(n_neighbors = 18)), 
                ('mlp0', MLPClassifier(hidden_layer_sizes=(15, 15))),
                ('mlp4', MLPClassifier(hidden_layer_sizes=(20, 20))),
                ('RFclf1', RandomForestClassifier())], 
    voting='soft')
voteclf2 = voteclf2.fit(X, y)
voteclf2.score(X_test, y_test)
voteclf3 = VotingClassifier(
    estimators=[('svcclf0', SVC()),
                ('knn0', KNeighborsClassifier(n_neighbors = 18)),
                ('RFclf03', RandomForestClassifier(n_estimators = 20)), 
                ('mlp0', MLPClassifier(hidden_layer_sizes=(15, 15))),
                ('mlp4', MLPClassifier(hidden_layer_sizes=(20, 20)))],
                 
    voting='hard')
voteclf3 = voteclf3.fit(X, y)
voteclf3.score(X_test, y_test)
cm = confusion_matrix(y_test, voteclf2.predict(X_test), labels = class_names)
sns.heatmap(cm, annot = True, fmt = "d", xticklabels = class_names, yticklabels = class_names)