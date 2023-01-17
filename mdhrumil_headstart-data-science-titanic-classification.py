import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import seaborn as sns
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
data = pd.concat([train,test],axis=0,sort=False)
data.head()
train.info()
train[['Pclass','Survived']].groupby(train['Pclass']).mean()
train[['Sex','Survived']].groupby(train['Sex']).mean()
for row in train:
    train['Family size'] = train['SibSp'] + train['Parch'] + 1

for row in test:
    test['Family size'] = test['SibSp'] + test['Parch'] + 1
train[['Family size','Survived']].groupby(train['Family size']).mean()
for row in train:
        train['isAlone'] = 0
        train.loc[train['Family size']==1, 'isAlone'] = 1

for row in test:
        test['isAlone'] = 0
        test.loc[test['Family size']==1, 'isAlone'] = 1
        
train[['isAlone','Survived']].groupby(train['isAlone']).mean()
data['Embarked'].groupby(data['Embarked']).count()
train['Embarked'] = train['Embarked'].fillna('S')
train[['Embarked','Survived']].groupby(train['Embarked']).mean()

test['Embarked'] = test['Embarked'].fillna('S')
train[['Embarked','Survived']].groupby(train['Embarked']).mean()
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'],4)

test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['CategoricalFare'] = pd.qcut(test['Fare'],4)
train[['CategoricalFare','Survived']].groupby(train['CategoricalFare']).mean()
data.info()
for row in train:
    avg = train['Age'].mean()
    std = train['Age'].std()
    null_count = train['Age'].isnull().sum()
    age_null_random_list = np.random.randint(avg - std, avg + std, size=null_count)
    train['Age'][np.isnan(train['Age'])] = age_null_random_list
    train['Age'] = train['Age'].astype(int)

for row in test:
    avg = test['Age'].mean()
    std = test['Age'].std()
    null_count = test['Age'].isnull().sum()
    age_null_random_list = np.random.randint(avg - std, avg + std, size=null_count)
    test['Age'][np.isnan(test['Age'])] = age_null_random_list
    test['Age'] = test['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'],5)

test['CategoricalAge'] = pd.cut(test['Age'],5)
train[['CategoricalAge','Survived']].groupby(train['CategoricalAge']).mean()
 # Mapping Fare
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3

train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4


test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[ test['Fare'] > 31, 'Fare'] = 3

test.loc[ test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age'] = 4

train.describe()
test.describe()
train.head()
train['Sex'] = train['Sex'].map({'female':0,'male':1})
test['Sex'] = test['Sex'].map({'female':0,'male':1})
train['Embarked'] = train['Embarked'].map({'S':0,'C':1,'Q':2})
test['Embarked'] = test['Embarked'].map({'S':0,'C':1,'Q':2})
train = train.drop(['Name','SibSp','Parch','Ticket','Cabin','CategoricalFare','CategoricalAge'],axis = 1)
test = test.drop(['Name','SibSp','Parch','Ticket','Cabin','CategoricalFare','CategoricalAge'], axis = 1)
sns.pairplot(train,hue='Survived')
sns.countplot(x=train['Sex'],data=train,hue="Survived",orient='v')
plt.figure(figsize=(12,10))
sns.countplot(y=train['Age'],data=train,hue='Sex')
plt.figure(figsize=(15,15))
sns.jointplot(train['Embarked'],train['Pclass'],kind="kde")
plt.figure(figsize=(12,10))
sns.countplot(x=train['Pclass'],data=train,hue="Fare") 
explode =(0,0.05,0.05,0.05)
plt.figure(figsize=(10,10))
plt.pie(train['Fare'].groupby(train['Fare']).sum(),labels=['Category 0','Category 1','Category 2','Category 3'],
        colors=['gold','#e33d3d','#33d9ed','#7ae10c'],
        explode=explode,shadow=True,autopct='%1.1f%%')
plt.show()
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

x = train.iloc[:,[0,2,3,4,5,6,7,8]].values
y = train.iloc[:,1].values

classifiers = [
    KNeighborsClassifier(4),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LogisticRegression()]

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

chart = pd.DataFrame(columns=["Classifier", "Accuracy"])

acc_dict = {}

for train_index, test_index in sss.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(x_train, y_train)
    train_predictions = clf.predict(x_test)
    acc = accuracy_score(y_test, train_predictions)
    if name in acc_dict:
        acc_dict[name] += acc
    else:
        acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=["Classifier", "Accuracy"])
    chart = chart.append(log_entry)
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=chart, palette="Blues")
chart['Accuracy']= chart['Accuracy']*1000
chart
final_classifier = LogisticRegression()
final_classifier.fit(train.iloc[:,[0,2,3,4,5,6,7,8]].values,train.iloc[:,1].values)
result = final_classifier.predict(test)
result = DataFrame(result)
result