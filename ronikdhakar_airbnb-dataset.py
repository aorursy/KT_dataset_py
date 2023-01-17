import pandas as pd
import numpy as np
import seaborn as sn
import scipy as sp
import graphviz
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn import neighbors
import matplotlib.patches as mpatches
from sklearn.tree import export_graphviz
%matplotlib inline


df=pd.read_csv("../input/airbnb.csv", na_values=["Not Available","NaN"])
df.set_index('id', inplace=True)
df.fillna(0, inplace=True)
df.head()
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

X, y = make_classification(n_samples=1000,n_features=8,n_informative=3,n_redundant=0,n_repeated=0,n_classes=2,random_state=0,
                           shuffle=False)

forest = ExtraTreesClassifier(n_estimators=250,random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
pt.figure()
pt.title("Feature importances")

pt.bar(range(X.shape[1]), importances[indices],color="r", yerr=std[indices])
pt.xticks(range(X.shape[1]), indices)
pt.xlim([-1, X.shape[1]])
pt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

X_airbnb, y_airbnb = make_blobs(n_samples = 100, centers = 8, cluster_std = 1.3,random_state = 4)
X_train, X_test, y_train, y_test = train_test_split(X_airbnb, y_airbnb, random_state = 0)

clf = RandomForestClassifier( random_state = 0)
clf.fit(X_train, y_train)

print('Airbnb dataset')
print('Accuracy of Random Forest classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
from sklearn.dummy import DummyClassifier

dummy_majority = DummyClassifier(strategy = 'uniform').fit(X_train, y_train)
y_dummy_predictions = dummy_majority.predict(X_test)

y_dummy_predictions
dummy_majority.score(X_test, y_test)
from sklearn.metrics import confusion_matrix

dummy_majority = DummyClassifier(strategy = 'uniform').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test, y_majority_predicted)

print('Uniform class (dummy classifier)\n', confusion)
import seaborn as sns

X_train_ab, X_test_ab, y_train_ab, y_test_ab = train_test_split(X_airbnb, y_airbnb, random_state=0)

clf = RandomForestClassifier().fit(X_train_ab, y_train_ab)
clf_predicted_ab = clf.predict(X_test_ab)
confusion_ab = confusion_matrix(y_test_ab, clf_predicted_ab)
df_ab = pd.DataFrame(confusion_ab,index = [i for i in range(0,8)], columns = [i for i in range(0,8)])
df_ab

pt.figure(figsize=(5.5,4))
sns.heatmap(df_ab, annot=True)

pt.ylabel('True label')
pt.xlabel('Predicted label')

from sklearn.metrics import classification_report

print(classification_report(y_test_ab, clf_predicted_ab))