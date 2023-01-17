import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.isnull().sum(axis=0)
sns.countplot(dataset['Embarked'])
sns.barplot('Pclass', 'Survived', data = train)
train = train.fillna({"Embarked": "S"})
train = pd.get_dummies(train, columns = ['Sex'])
train = pd.get_dummies(train, columns = ['Embarked'])
train.head()
data = [train, test]
for dataset in data:
    mean = train["Age"].mean()
    std = test["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train["Age"].astype(int)
train.drop('Cabin',axis = 1,inplace = True)
train.drop('Ticket',axis = 1,inplace = True)
train.drop('Name',axis = 1,inplace = True)
train.drop('PassengerId',axis = 1,inplace = True)
train_c = train[['Survived']]
train_f = train.drop(['Survived'], axis = 1)
train_f.head()
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
X_train, X_test, Y_train, Y_test = train_test_split(train_f, train_c, test_size = 0.20,random_state = 42)
clf = DecisionTreeClassifier(criterion = 'entropy')
clf.fit(X_train, Y_train)
export_graphviz(clf, out_file = 'Tree.dot')

pred = clf.predict(X_test)
hit_rate = accuracy_score(Y_test, pred)
error_rate = 1 - hit_rate
hit_rate * 100
print(classification_report(Y_test,pred))
print(confusion_matrix(Y_test,pred))
sub = pd.DataFrame({ "PassengerId": test['PassengerId'], "Survived": pred})

sub.head()

sub.to_csv('MySub.csv', index = False)