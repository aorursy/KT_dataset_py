# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

sonar=pd.read_csv("/kaggle/input/sonardata/sonar.csv")
sonar.head(5)
sonar.describe()
df = pd.DataFrame()
from sklearn.model_selection import train_test_split
X=sonar.iloc[:,0:60]
y=sonar.iloc[:,60]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#Initalize the classifier


log_clf = LogisticRegression(random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=10)
svm_clf = SVC(gamma="auto", random_state=42, probability=True)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('knn', knn_clf), ('svc', svm_clf)],
    voting='hard')

voting_clf_soft = VotingClassifier(
    estimators=[('lr', log_clf), ('knn', knn_clf), ('svc', svm_clf)],
    voting='soft')

# Voting can be chnaged to 'Soft', however the classifer must support predict probability
voting_clf.fit(X_train, y_train)
voting_clf_soft.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

for clf in (log_clf, knn_clf, svm_clf, voting_clf,voting_clf_soft):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    a_row = pd.Series([clf.__class__.__name__, accuracy_score(y_test, y_pred)])
    row_df = pd.DataFrame([a_row])
    df = pd.concat([row_df, df], ignore_index=False)
df.iat[0,0]='Soft'
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))
a_row = pd.Series([tree_clf.__class__.__name__, accuracy_score(y_test, y_pred_tree)])
row_df = pd.DataFrame([a_row])
df = pd.concat([row_df, df], ignore_index=True)
from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=120, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
a_row = pd.Series([bag_clf.__class__.__name__, accuracy_score(y_test, y_pred)])
row_df = pd.DataFrame([a_row])
df = pd.concat([row_df, df], ignore_index=True)
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))
a_row = pd.Series([rnd_clf.__class__.__name__, accuracy_score(y_test, y_pred_rf)])
row_df = pd.DataFrame([a_row])
df = pd.concat([row_df, df], ignore_index=True)
nc=np.arange(5,60,5)
acc=np.empty(11)
i=0
for k in np.nditer(nc):
    rnd_clf=RandomForestClassifier(n_estimators=500, max_leaf_nodes=int(k), n_jobs=-1, random_state=42)
    rnd_clf.fit(X_train, y_train)
    acc[i]=rnd_clf.score(X_test, y_test)
    i = i + 1
acc
x=pd.Series(acc,index=nc)
x.plot()
# Add title and axis names
plt.title('Random Forest No of leaves vs Accuracy')
plt.xlabel('Numer of Trees')
plt.ylabel('Accuracy')
plt.show()
df.columns=['Method','Accuracy']
from itertools import cycle, islice
my_colors = list(islice(cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k']), None, len(df)))
df.plot.barh(x='Method', y='Accuracy', rot=0,color=my_colors,figsize=(15,8))