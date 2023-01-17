# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/train.csv")
dataset.head(5)
dataset.describe()
dataset.info()
dataset.info()
dataset["Age"][dataset["Sex"]=="female"].median()
dataset["Age"][dataset["Sex"]=="male"].median()
dataset["Age"][dataset["Sex"]=="female"]=dataset["Age"].fillna(dataset["Age"][dataset["Sex"]=="female"].median())
dataset["Age"][dataset["Sex"]=="male"]=dataset["Age"].fillna(dataset["Age"][dataset["Sex"]=="male"].median())
dataset.info()
dataset.head(5)
dataset = dataset.drop(["PassengerId","Name","Cabin","Ticket"],axis=1)
dataset.head(5)
dataset["Sex"]=dataset["Sex"].replace("male",1)

dataset["Sex"]=dataset["Sex"].replace("female",0)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

LE = LabelEncoder()
dataset['Embarked']=dataset['Embarked'].fillna(value='S')
dataset['Embarked']= LE.fit_transform(dataset['Embarked'])
dataset=pd.get_dummies(dataset,columns=['Pclass',"Embarked"],drop_first=True)



dataset.head(5)
X = dataset.iloc[:,1:]

Y = dataset.iloc[:,0]
from sklearn.preprocessing import MinMaxScaler

norm = MinMaxScaler()

N_X=norm.fit_transform(X)
N_X
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
k_range = list(range(1,20))

weight = ["uniform","distance"]

param_grid = dict(n_neighbors=k_range,weights = weight)

grid =GridSearchCV(knn,param_grid,cv=10,scoring="accuracy",return_train_score=False)

grid.fit(N_X,Y)
print(grid.best_score_)

print(grid.best_params_)
test= pd.read_csv("../input/test.csv")
test1= pd.read_csv("../input/test.csv")
test.head(5)
test.describe()
test.info()
test["Age"][test["Sex"]=="female"]=test["Age"].fillna(test["Age"][test["Sex"]=="female"].median())
test["Age"][test["Sex"]=="male"]=test["Age"].fillna(test["Age"][test["Sex"]=="male"].median())
test = test.drop(["PassengerId","Name","Cabin","Ticket"],axis=1)
test.info()
test["Sex"]=test["Sex"].replace("male",1)

test["Sex"]=test["Sex"].replace("female",0)
test["Fare"]=test["Fare"].fillna(test["Fare"].median())
test['Embarked']= LE.fit_transform(test['Embarked'])
test =pd.get_dummies(test,columns=['Pclass','Embarked'],drop_first=True)

test.head(5)
normt = MinMaxScaler()

t_x=norm.fit_transform(test)
pred = grid.predict(t_x)
data = pd.DataFrame(test1["PassengerId"],columns=["PassengerId"])
data["Survived"]=pred
data.head(5)
#data.to_csv('submission2.csv', index=False)
data.shape
data.columns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
logi_clf = LogisticRegression(random_state=0)

logi_parm = {"penalty": ['l1', 'l2'], "C": [0.1, 0.5, 1, 5, 10, 50]}



svm_clf = SVC(random_state=0)

svm_parm = {'kernel': ['rbf', 'poly'], 'C': [0.1, 0.5, 1, 5, 10, 50], 'degree': [3, 5, 7], 

            'gamma': ['auto', 'scale']}



dt_clf = DecisionTreeClassifier(random_state=0)

dt_parm = {'criterion':['gini', 'entropy']}



knn_clf = KNeighborsClassifier()

knn_parm = {'n_neighbors':[5, 10, 15, 20], 'weights':['uniform', 'distance'], 'p': [1,2]}



#gnb_clf = GaussianNB()

#gnb_parm = {'priors':['None']}



clfs = [logi_clf, svm_clf, dt_clf, knn_clf]#, gnb_clf]

params = [logi_parm, svm_parm, dt_parm, knn_parm]#, gnb_parm] 
clfs_opt = []

clfs_best_scores = []

clfs_best_param = []

for clf_, param in zip(clfs, params):

    clf = RandomizedSearchCV(clf_, param, cv=5)

    clf.fit(N_X, Y)

    clfs_opt.append(clf)

    clfs_best_scores.append(clf.best_score_)

    clfs_best_param.append(clf.best_params_)
max(clfs_best_scores)
arg = np.argmax(clfs_best_scores)

clfs_best_param[arg]
clf = clfs_opt[arg]
pred1 = clf.predict(t_x)
data1 = pd.DataFrame(test1["PassengerId"],columns=["PassengerId"])

data1["Survived"]=pred1
#data1.to_csv('submission2.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.svm import SVC 

from sklearn.ensemble import VotingClassifier 

from sklearn.model_selection import cross_val_score
clf1 = KNeighborsClassifier(n_neighbors=6,weights='uniform')

clf2 =SVC(kernel='rbf', gamma='auto_deprecated', degree=5, C=10, probability=True)

clf3 = VotingClassifier(estimators=[("svm",clf2),("knn",clf1)],voting='hard')
clf1 = clf1.fit(N_X,Y)

clf2 = clf2.fit(N_X,Y)

clf3 = clf3.fit(N_X,Y)
for clf , label in zip ([clf1,clf2,clf3,clf4],["knn","svm","hard","soft"]):

    score = cross_val_score(clf,N_X,Y,cv=6,scoring="accuracy")

    print("accuracy: %0.2f (+/- %0.2f)[%s]"%(score.mean(),score.std(),label))

clf4 = VotingClassifier(estimators=[("svm",clf2),("knn",clf1)],voting='soft',weights=[4,1])
clf11 = clf1.fit(N_X,Y)

clf22 = clf2.fit(N_X,Y)

clf33 = clf4.fit(N_X,Y)
predict =clf33.predict(t_x)
data_estimaor = pd.DataFrame(test1["PassengerId"],columns=["PassengerId"])

data_estimaor["Survived"]=predict
data_estimaor.to_csv('submission-est.csv', index=False)