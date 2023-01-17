# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, precision_score

from sklearn.metrics import recall_score, f1_score, accuracy_score

from sklearn.preprocessing import OneHotEncoder



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train = train.drop(["Name","Ticket","Cabin"],axis=1)

train.head()

age = train.groupby(['Sex', 'Pclass'])['Age'].agg(['mean', 'median']).round(1)

train_null = train[train["Age"].isnull()]

age

for x in range(train.shape[0]):

    if(pd.isnull(train.iloc[x,4])):

        train.iloc[x,4] = age.loc[train.iloc[x,3],train.iloc[x,2]][0].astype('int')

train.isnull().sum()
sns.set(color_codes=True)

sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=train);
sns.distplot(train["Fare"], kde=False, rug=True);
sns.distplot(train["Age"], kde=False, rug=True);
sns.heatmap(train.corr(), annot=True)

#plt.tight_layout()
train = pd.read_csv("/kaggle/input/titanic/train.csv")



age = train.groupby(['Sex', 'Pclass'])['Age'].agg(['mean', 'median']).round(1)

for x in range(train.shape[0]):

    if(pd.isnull(train.iloc[x,5])):

        train.iloc[x,5] = age.loc[train.iloc[x,4],train.iloc[x,2]][0].astype('int')



train['Family']=train["SibSp"]+train["Parch"]+1

train['Fam_type'] = pd.cut(train.Family, [0,1,4,7,11], labels=[1,2,3,4])



train['Title'] = train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

train['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)

train['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)





train = train.drop(["Name","Ticket","Cabin","PassengerId","SibSp","Parch","Family"],axis=1)

train.loc[train.isnull().loc[:,"Embarked"],"Embarked"] = "S"

train["Fare"] = pd.qcut(train["Fare"], 5, labels=[1,2,3,4,5]).astype(int)

train["Age"] = pd.qcut(train["Age"], 5, labels=[1,2,3,4,5]).astype(int)

train = pd.get_dummies(train, columns=["Sex","Pclass","Embarked","Title"])

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

passangersID = test["PassengerId"]



age = test.groupby(['Sex', 'Pclass'])['Age'].agg(['mean', 'median']).round(1)

for x in range(test.shape[0]):

    if(pd.isnull(test.iloc[x,4])):

        test.iloc[x,4] = age.loc[test.iloc[x,3],test.iloc[x,1]][0].astype('int')



test['Family']=test["SibSp"]+test["Parch"]+1

test['Fam_type'] = pd.cut(test.Family, [0,1,4,7,11], labels=[1,2,3,4])



test['Title'] = test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

test['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)

test['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)



test.loc[test.isnull().loc[:,"Embarked"],"Embarked"] = "S"

test.loc[test.isnull().loc[:,"Fare"],"Fare"] = 7.75 # mode



test["Fare"] = pd.qcut(test["Fare"], 5, labels=[1,2,3,4,5]).astype(int)

test["Age"] = pd.qcut(test["Age"], 5, labels=[1,2,3,4,5]).astype(int)



test = pd.get_dummies(test, columns=["Sex","Pclass","Embarked","Title"])



test = test.drop(["Name","Ticket","Cabin","PassengerId","SibSp","Parch","Family"],axis=1)

test.head()
y = train["Survived"]

X = train.drop("Survived",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_test2= test
#scaler = MinMaxScaler()

#X_train = scaler.fit_transform(X_train)

#X_test = scaler.transform(X_test)

#X_test2 = scaler.transform(X_test2)
knn = KNeighborsClassifier(n_neighbors = 6)

knn.fit(X_train, y_train)

knn.score(X_test, y_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import classification_report

prediction = knn.predict(X_test)



# Accuracy = TP + TN / (TP + TN + FP + FN)

# Precision = TP / (TP + FP)

# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate

# F1 = 2 * Precision * Recall / (Precision + Recall) 

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, prediction)))

print('Precision: {:.2f}'.format(precision_score(y_test, prediction)))

print('Recall: {:.2f}'.format(recall_score(y_test, prediction)))

print('F1: {:.2f}'.format(f1_score(y_test, prediction)))

print(classification_report(y_test, prediction, target_names=['dead', 'survived']))
prediction = knn.predict(X_test2)

output = pd.DataFrame({'PassengerId': passangersID, 'Survived': prediction})
output.to_csv('KNeigborsClassifier.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=0.1).fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
prediction = clf.predict(X_test2)

output = pd.DataFrame({'PassengerId': passangersID, 'Survived': prediction})
output.to_csv('LogicalRegression.csv', index=False)

print("Your submission was successfully saved!")
#poly = PolynomialFeatures(degree=2)

#X_train, X_test, y_train, y_test = train_test_split(X, y)

#X_train_scaled = scaler.fit_transform(X_train)

#X_test_scaled = scaler.transform(X_test)

#X_train_scaled_poly = poly.fit_transform(X_train_scaled)

#X_test_scaled_poly = poly.fit_transform(X_test_scaled)
#z = list()

#for name in names1:

#    z.append(name.split(",")[1].split(".")[0])

#t = pd.DataFrame(z)

#t = t.astype('category')

#train["Name"] = t

#train.head()
parameters = {'criterion': ['entropy', 'gini'],

              'min_samples_split': [5*x for x in range(1,15,2)],

              'min_samples_leaf': [2*x+1 for x in range(14)],

              'max_leaf_nodes': [2*x for x in range(1, 9)],

              'max_depth': [2*x for x in range(1,9)]}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=parameters, cv=3)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
def calculate_metrics(model, X_test, y_test):

    pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred)

    acc = accuracy_score(y_test, pred)

    precision = precision_score(y_test, pred)

    recall = recall_score(y_test, pred)

    f_score = f1_score(y_test, pred)

    print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1_score: {}'.format(

        acc, precision, recall, f_score))

    return cm
best_model = DecisionTreeClassifier(**grid_search.best_params_)

best_model.fit(X_train, y_train)



cm = calculate_metrics(best_model, X_test, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label');
clf = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
prediction = clf.predict(X_test2)

output = pd.DataFrame({'PassengerId': passangersID, 'Survived': prediction})
output.to_csv('DecisionTree.csv', index=False)

print("Your submission was successfully saved!")
clf = SVC(C=0.1).fit(X_train, y_train)

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))
clf = SVC(kernel = 'poly', degree = 3).fit(X_train, y_train)

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))
best = list()

best.append(0)

for mygamma in [0.1 , 1 , 10, 100]:

    for myC in [0.1,1 , 10 , 100]:

        clf = SVC(kernel = 'rbf', gamma=mygamma,C=myC).fit(X_train, y_train)

        print("gamma: " +str(mygamma)+", C: "+str(myC))

        print(clf.score(X_train,y_train))

        print(clf.score(X_test,y_test))

        
clf = SVC(kernel = 'rbf',C=1,gamma=0.1).fit(X_train, y_train)

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))
prediction = clf.predict(X_test2)

output = pd.DataFrame({'PassengerId': passangersID, 'Survived': prediction})

output.to_csv('KernelizedSVM.csv', index=False)

print("Your submission was successfully saved!")
clf = RandomForestClassifier(max_depth=4,n_estimators=10000).fit(X_train, y_train)

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))
prediction = clf.predict(X_test2)

output = pd.DataFrame({'PassengerId': passangersID, 'Survived': prediction})

output.to_csv('RandomTreeClassifier.csv', index=False)

print("Your submission was successfully saved!")
gbrt = GradientBoostingClassifier(n_estimators=700,max_depth=3,learning_rate=0.004).fit(X_train, y_train)

print(gbrt.score(X_train,y_train))

print(gbrt.score(X_test,y_test))
prediction = gbrt.predict(X_test2)

output = pd.DataFrame({'PassengerId': passangersID, 'Survived': prediction})

output.to_csv('GradientBoostingClassifier.csv', index=False)

print("Your submission was successfully saved!")
LogicalRegression = pd.read_csv("/kaggle/working/LogicalRegression.csv")

DecisionTree = pd.read_csv("/kaggle/working/DecisionTree.csv")

KernelizedSVM = pd.read_csv("/kaggle/working/KernelizedSVM.csv")

RandomTreeClassifier = pd.read_csv("/kaggle/working/RandomTreeClassifier.csv") 

GradientBoostingClassifier = pd.read_csv("/kaggle/working/GradientBoostingClassifier.csv")
output = LogicalRegression

output["sum"] = LogicalRegression["Survived"] +DecisionTree["Survived"]+KernelizedSVM["Survived"] 

+RandomTreeClassifier["Survived"] +GradientBoostingClassifier["Survived"]



output["prediction"] =0

output.loc[output["sum"]>2,"prediction"] =1

output = output.drop(["sum"],axis=1)

output["Survived"] = output["prediction"]

output = output.drop("prediction",axis=1)
output.to_csv('MajorityVote.csv', index=False)

print("Your submission was successfully saved!")