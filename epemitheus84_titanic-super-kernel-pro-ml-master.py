import numpy as np 
import pandas as pd

import os
# print files 
print(os.listdir("../input"))
train_data_set = pd.read_csv('../input/train.csv')
train_data_set
import matplotlib.pyplot as plt

males = train_data_set[train_data_set["Sex"] == 'male']['Sex'].count()
females = train_data_set[train_data_set["Sex"] == 'female']['Sex'].count()
plt.pie([males, females], labels=['Male', 'Female'])
import matplotlib.pyplot as plt

males = train_data_set[train_data_set["Survived"] == 1]['Survived'].count()
females = train_data_set[train_data_set["Survived"] == 0]['Survived'].count()
plt.pie([males, females], labels=['Survive', 'Not survive'])
import matplotlib.pyplot as plt
unique_values = train_data_set["Pclass"].unique()
values = []
for value in unique_values:
    values.append(train_data_set[train_data_set["Pclass"] == value]['Pclass'].count())
    
plt.pie(values, labels=unique_values)
import matplotlib.pyplot as plt
unique_values = train_data_set["SibSp"].unique()
values = []
for value in unique_values:
    values.append(train_data_set[train_data_set["SibSp"] == value]['SibSp'].count())
    
plt.pie(values, labels=unique_values)
import matplotlib.pyplot as plt
unique_values = train_data_set["Parch"].unique()
values = []
for value in unique_values:
    values.append(train_data_set[train_data_set["Parch"] == value]['Parch'].count())
    
plt.pie(values, labels=unique_values)
seed = 0
X = train_data_set.iloc[:,[2,4,5,6,7,9]]
y = train_data_set.iloc[:, 1]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X.iloc[:,1] = le.fit_transform(X.iloc[:,1])
X
X = X.fillna(X.mean())
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=seed, solver='lbfgs').fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy_score(y_test, y_pred)
import itertools

def beauty_cm(cm):
    labels = ['Predicted NO', 'Predicted YES','Actual NO','Actual YES']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier \n')

    ax.set_xticklabels([''] + labels[0:2])
    ax.set_yticklabels([''] + labels[2:4])

    fmt = '.0f'

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="red", fontsize = 22)

    plt.show()

beauty_cm(confusion_matrix(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=31).fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
beauty_cm(confusion_matrix(y_test, y_pred))
from sklearn import svm
clf = svm.SVC(random_state=seed, gamma='scale').fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
beauty_cm(confusion_matrix(y_test, y_pred))
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
beauty_cm(confusion_matrix(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=seed, min_samples_split=5, min_samples_leaf=8).fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
beauty_cm(confusion_matrix(y_test, y_pred))
from sklearn import tree 
from sklearn.externals.six import StringIO  
import pydot 
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_png("dtr.png") 
from IPython.display import Image
Image(filename='dtr.png') 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1111, min_samples_split=5, min_samples_leaf=8, random_state=seed)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
beauty_cm(confusion_matrix(y_test, y_pred))
dot_data = StringIO() 
estimator = clf.estimators_[0]
tree.export_graphviz(estimator, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_png("rf.png") 
from IPython.display import Image
Image(filename='rf.png') 
test_data_set = pd.read_csv('../input/test.csv')
test_data_set.head()
X = test_data_set.iloc[:,[1,3,4,5,6,8]]
le = preprocessing.LabelEncoder()
X.iloc[:,1] = le.fit_transform(X.iloc[:,1])
X = X.fillna(X.mean())
X
y_pred = clf.predict(X)
y_pred
ids = pd.DataFrame({'PassengerId':range(892, len(y_pred)+1)})
results = ids.assign(Survived = y_pred)
results