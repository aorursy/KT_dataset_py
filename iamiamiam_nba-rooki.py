#Author Mina Samizade
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import os
print(os.listdir("../input"))
dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')

Y = np.array(dataset.get('TARGET_5Yrs'))
dataset = dataset.drop(['PlayerID','Name','TARGET_5Yrs'] ,axis=1)
dataset = dataset.fillna(0)
testset = testset.drop(['PlayerID','Name'] ,axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, Y, stratify = Y, test_size=.2, random_state = 12)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
numberOfFeatures = 10
#X_new = SelectKBest(chi2, k=numberOfFeatures).fit_transform(dataset["X_train"], dataset["y_train"])
#print(X_train.shape)
#print(X_test.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=60)
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)
knn_score = accuracy_score(knn_prediction,y_test)
print('KNN: ', accuracy_score(knn_prediction,y_test))
from sklearn.svm import SVC

svc0 = SVC(C=1, cache_size=400, class_weight=None, coef0=0.01,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=False,
    tol=0.0128, verbose=False)
svc0.fit(X_train, y_train)
svc0_prediction = svc0.predict(X_test)
svc0_score = accuracy_score(svc0_prediction,y_test)
print('SVC0: ', accuracy_score(svc0_prediction,y_test))

svc1 = SVC(kernel="linear", C=1, gamma='auto')
svc1.fit(X_train, y_train)
svc1_prediction = svc1.predict(X_test)
svc1_score = accuracy_score(svc1_prediction,y_test)
print('SVC1: ', accuracy_score(svc1_prediction,y_test))

svc2 = SVC(gamma=4, C=1, probability=True)
svc2.fit(X_train, y_train)
svc2_prediction = svc2.predict(X_test)
svc2_score = accuracy_score(svc2_prediction,y_test)
print('SVC2: ', accuracy_score(svc2_prediction,y_test))

svc3 = SVC(C=5, gamma='auto')
svc3.fit(X_train, y_train)
svc3_prediction = svc3.predict(X_test)
svc3_score = accuracy_score(svc3_prediction,y_test)
print('SVC3: ', accuracy_score(svc3_prediction,y_test))
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore")

svc4 = LinearSVC(C=1, random_state=0)
svc4.fit(X_train, y_train)
svc4_prediction = svc4.predict(X_test)
svc4_score = accuracy_score(svc4_prediction,y_test)
print('SVC4: ', accuracy_score(svc4_prediction,y_test))

from sklearn.neural_network import MLPClassifier

mlp1 = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(12, 5), random_state=1)
mlp1.fit(X_train, y_train)
mlp1_prediction = mlp1.predict(X_test)
mlp1_score = accuracy_score(mlp1_prediction,y_test)
print('mlp1: ', accuracy_score(mlp1_prediction,y_test))

from sklearn.tree import DecisionTreeClassifier

tree =  DecisionTreeClassifier(max_depth=2)
tree.fit(X_train, y_train)
tree_prediction = tree.predict(X_test)
tree_score = accuracy_score(tree_prediction,y_test)
print('tree: ', accuracy_score(tree_prediction,y_test))
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(random_state=2)
lg.fit(X_train, y_train)
lg_prediction = lg.predict(X_test)
lg_score = accuracy_score(lg_prediction,y_test)
print('lg: ', accuracy_score(lg_prediction,y_test))
from sklearn.ensemble import RandomForestClassifier

randomForest = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=2)
randomForest.fit(X_train, y_train)
randomForest_prediction = randomForest.predict(X_test)
randomForest_score = accuracy_score(randomForest_prediction,y_test)
print('randomForest: ', accuracy_score(randomForest_prediction,y_test))

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=100)
ada.fit(X_train, y_train)
ada_prediction = ada.predict(X_test)
ada_score = accuracy_score(ada_prediction, y_test)
print("AdaBoost: ", accuracy_score(ada_prediction, y_test))
from sklearn.ensemble import GradientBoostingClassifier

GBDT = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=1, random_state = 0)
GBDT.fit(X_train, y_train)
GBDT_prediction = GBDT.predict(X_test)
GBDT_score = accuracy_score(GBDT_prediction, y_test)
print("GBDT: " , accuracy_score(GBDT_prediction, y_test))
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings("ignore")

VC1 = VotingClassifier(estimators=[('tree',tree),('knn', knn), ('lg',lg), ('RandomForestClassifier',randomForest), ('GBDT',GBDT)], 
                         voting='soft', weights=[1,3,1,3,2], flatten_transform=True)
VC1.fit(X_train, y_train)
VC1_prediction = VC1.predict(X_test)
print("VC1: " , accuracy_score(VC1_prediction, y_test))
VC2 = VotingClassifier(estimators=[('VC1', VC1),('SCV1',svc1),('SVC4',svc4), ('knn', knn), ('RandomForestClassifier',randomForest), ('GBDT',GBDT)], 
                         voting='hard', weights=[3,1.2,1.5,2.5,2,2], flatten_transform=True)
VC2.fit(X_train, y_train)
VC2_prediction = VC2.predict(X_test)
print("VC2: " , accuracy_score(VC2_prediction, y_test))
VC3 = VotingClassifier(estimators=[('VC2', VC2), ('VC1', VC1),('SCV1',svc1),('SVC4',svc4), ('RandomForestClassifier',randomForest), ('GBDT',GBDT)], 
                         voting='hard', weights=[3,2,1.2,1.5,3,2], flatten_transform=True)
VC3.fit(X_train, y_train)
VC3_prediction = VC3.predict(X_test)
print("VC3: " , accuracy_score(VC3_prediction, y_test))

mm = VC3.predict(testset)

cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': [x for x in mm] }
submission = pd.DataFrame(cols)
#print(submission)
submission.to_csv("submission.csv", index=False)
