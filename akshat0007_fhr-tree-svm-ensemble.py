import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/CTG.csv")
df.head()
df=df.drop(["FileName","Date","SegFile","b","e"],axis=1)
df.head()
df.columns
df.shape
df.isnull().sum()
df=df.dropna()
df.isnull().sum()
df.dtypes
X=df[['LBE', 'LB', 'AC', 'FM', 'UC', 'DL',

       'DS', 'DP', 'DR']]

Y=df[["NSP"]]
from sklearn.preprocessing import StandardScaler, MinMaxScaler

Scaler=StandardScaler()

X=Scaler.fit_transform(X)



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.svm import SVC



svm_clf=SVC(kernel="poly",degree=6,coef0=5,gamma=0.1)

svm_clf=svm_clf.fit(X_train,y_train)

y_pred=svm_clf.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
print(confusion_matrix(y_test,y_pred))
f1_score(y_test,y_pred,average='weighted')


accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred,average='weighted')
recall_score(y_test,y_pred,average="weighted")
from sklearn.tree import DecisionTreeClassifier
tree_clf=DecisionTreeClassifier(min_samples_split=6, min_samples_leaf=4, max_depth=6, )

tree_clf=tree_clf.fit(X_train,y_train)

y_pred=tree_clf.predict(X_test)
accuracy_score(y_test,y_pred)
recall_score(y_test,y_pred,average="weighted")
precision_score(y_test,y_pred,average='weighted')
from sklearn.tree import export_graphviz

export_graphviz(

tree_clf, out_file="tree.dot",

feature_names=['LBE', 'LB', 'AC', 'FM', 'UC', 'DL',

       'DS', 'DP', 'DR'],

class_names="NSP",

rounded=True,

filled=True)
from subprocess import check_call

check_call(['dot','-Tpng','tree.dot','-o','tree.png'])
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
svm_clf=SVC(kernel="poly",degree=6,coef0=5,gamma=0.1,probability=True)

decision_tree=DecisionTreeClassifier(min_samples_split=6, min_samples_leaf=4, max_depth=6)

rnd_clf=RandomForestClassifier()

voting_clf=VotingClassifier(estimators=[("svm",svm_clf),('rf',rnd_clf),("decision_tree",decision_tree)],voting="hard")
voting_clf.fit(X_train,y_train)
for clf in (rnd_clf, svm_clf,decision_tree, voting_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
from sklearn.ensemble import BaggingClassifier
bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,n_jobs=-1,max_samples=100, bootstrap=True)
bag_clf.fit(X_train,y_train)

y_pred=bag_clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
##This accuracy is better than our previous decision tree model

##Therefore, we will again call ensemble technique.
voting_clf=VotingClassifier(estimators=[("svm",svm_clf),('rf',rnd_clf),("bagging_clf",bag_clf)],voting="hard")
for clf in (rnd_clf, svm_clf,bag_clf, voting_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
##Using trial and hit method, and performing out-of-box evaluation

##Since a predictor never sees the oob instances during training, it can be evaluated on these instances,

##without the need for a separate validation set or cross-validation. You can evaluate the ensemble itself by

##averaging out the oob evaluations of each predictor.
bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,n_jobs=-1,max_samples=100, bootstrap=True,oob_score=True)

bag_clf.fit(X_train,y_train)

y_pred=bag_clf.predict(X_test)

print(bag_clf.oob_score_)

print(accuracy_score(y_test,y_pred))
from sklearn.ensemble import AdaBoostClassifier
ada_clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=1000,learning_rate=0.1)
ada_clf.fit(X_train,y_train)

y_pred=ada_clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
from xgboost import XGBClassifier

xgb_clf=XGBClassifier(learning_rate=0.001)

xgb_clf.fit(X_train,y_train)

y_pred=xgb_clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
voting_clf=VotingClassifier(estimators=[("svm",svm_clf),("xgb_clf",xgb_clf)],voting="hard")

for clf in (svm_clf,xgb_clf, voting_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))