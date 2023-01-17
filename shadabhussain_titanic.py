# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
titanic_train_data = pd.read_csv("../input/train.csv")
titanic_train_data.head()
titanic_train_data.isnull().sum()
titanic_train_data.describe()
f,ax = plt.subplots(3,4,figsize=(20,16))
sns.countplot('Pclass',data=titanic_train_data,ax=ax[0,0])
sns.countplot('Sex',data=titanic_train_data,ax=ax[0,1])
sns.boxplot(x='Pclass',y='Age',data=titanic_train_data,ax=ax[0,2])
sns.countplot('SibSp',hue='Survived',data=titanic_train_data,ax=ax[0,3],palette='husl')
sns.distplot(titanic_train_data['Fare'].dropna(),ax=ax[2,0],kde=False,color='b')
sns.countplot('Embarked',data=titanic_train_data,ax=ax[2,2])

sns.countplot('Pclass',hue='Survived',data=titanic_train_data,ax=ax[1,0],palette='husl')
sns.countplot('Sex',hue='Survived',data=titanic_train_data,ax=ax[1,1],palette='husl')
sns.distplot(titanic_train_data[titanic_train_data['Survived']==0]['Age'].dropna(),ax=ax[1,2],kde=False,color='r',bins=5)
sns.distplot(titanic_train_data[titanic_train_data['Survived']==1]['Age'].dropna(),ax=ax[1,2],kde=False,color='g',bins=5)
sns.countplot('Parch',hue='Survived',data=titanic_train_data,ax=ax[1,3],palette='husl')
sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=titanic_train_data,palette='husl',ax=ax[2,1])
sns.countplot('Embarked',hue='Survived',data=titanic_train_data,ax=ax[2,3],palette='husl')

ax[0,0].set_title('Total Passengers by Class')
ax[0,1].set_title('Total Passengers by Gender')
ax[0,2].set_title('Age Box Plot By Class')
ax[0,3].set_title('Survival Rate by SibSp')
ax[1,0].set_title('Survival Rate by Class')
ax[1,1].set_title('Survival Rate by Gender')
ax[1,2].set_title('Survival Rate by Age')
ax[1,3].set_title('Survival Rate by Parch')
ax[2,0].set_title('Fare Distribution')
ax[2,1].set_title('Survival Rate by Fare and Pclass')
ax[2,2].set_title('Total Passengers by Embarked')
ax[2,3].set_title('Survival Rate by Embarked')
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=titanic_train_data);
#plot distributions of age of passengers who survived or did not survive
a = sns.FacetGrid(titanic_train_data, hue = 'Survived', aspect=6 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , titanic_train_data['Age'].max()))
a.add_legend()
column = titanic_train_data.columns
column
# Sex
titanic_train_data.drop(['Ticket', 'Name'], inplace=True, axis=1)
titanic_train_data.Sex.fillna('0', inplace=True)
titanic_train_data.loc[titanic_train_data.Sex != 'male', 'Sex'] = 0
titanic_train_data.loc[titanic_train_data.Sex == 'male', 'Sex'] = 1
# Cabin
titanic_train_data.Cabin.fillna(0, inplace=True)
titanic_train_data.loc[titanic_train_data.Cabin.str[0] == 'A', 'Cabin'] = 1
titanic_train_data.loc[titanic_train_data.Cabin.str[0] == 'B', 'Cabin'] = 2
titanic_train_data.loc[titanic_train_data.Cabin.str[0] == 'C', 'Cabin'] = 3
titanic_train_data.loc[titanic_train_data.Cabin.str[0] == 'D', 'Cabin'] = 4
titanic_train_data.loc[titanic_train_data.Cabin.str[0] == 'E', 'Cabin'] = 5
titanic_train_data.loc[titanic_train_data.Cabin.str[0] == 'F', 'Cabin'] = 6
titanic_train_data.loc[titanic_train_data.Cabin.str[0] == 'G', 'Cabin'] = 7
titanic_train_data.loc[titanic_train_data.Cabin.str[0] == 'T', 'Cabin'] = 8
titanic_train_data.Cabin = titanic_train_data.Cabin.astype('int32')
sns.barplot(x = 'Cabin', y = 'Survived', order=[0,1,2,3,4,5,6,7,8], data=titanic_train_data)
# Embarked
titanic_train_data.loc[titanic_train_data.Embarked == 'C', 'Embarked'] = 1
titanic_train_data.loc[titanic_train_data.Embarked == 'Q', 'Embarked'] = 2
titanic_train_data.loc[titanic_train_data.Embarked == 'S', 'Embarked'] = 3
titanic_train_data.Embarked.fillna(0, inplace=True)
titanic_train_data.Age.fillna(titanic_train_data.Age.mean(), inplace=True)
titanic_train_data.isnull().sum()
grid = sns.FacetGrid(titanic_train_data, col='Survived', row='Pclass', size=2, aspect=3)
grid.map(plt.hist, 'Age', alpha=.5, bins=8)
grid.add_legend();
# Interactive chart using cufflinks
import cufflinks as cf
cf.go_offline()
titanic_train_data['Fare'].iplot(kind='hist', bins=30)
# Explore Fare distribution 
g = sns.distplot(titanic_train_data["Fare"], color="m", label="Skewness : %.2f"%(titanic_train_data["Fare"].skew()))
g = g.legend(loc="best")
# Corelation
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=titanic_train_data);
titanic_train_data['Id'] = titanic_train_data['PassengerId']
del titanic_train_data['PassengerId']
titanic_train_data.head()
# Corelation Heatmap
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(titanic_train_data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
titanic_test_data = pd.read_csv("../input/test.csv")
titanic_test_data.head()

# Sex
titanic_test_data.drop(['Ticket', 'Name'], inplace=True, axis=1)
titanic_test_data.Sex.fillna('0', inplace=True)
titanic_test_data.loc[titanic_test_data.Sex != 'male', 'Sex'] = 0
titanic_test_data.loc[titanic_test_data.Sex == 'male', 'Sex'] = 1

# Cabin
titanic_test_data.Cabin.fillna(0, inplace=True)
titanic_test_data.loc[titanic_test_data.Cabin.str[0] == 'A', 'Cabin'] = 1
titanic_test_data.loc[titanic_test_data.Cabin.str[0] == 'B', 'Cabin'] = 2
titanic_test_data.loc[titanic_test_data.Cabin.str[0] == 'C', 'Cabin'] = 3
titanic_test_data.loc[titanic_test_data.Cabin.str[0] == 'D', 'Cabin'] = 4
titanic_test_data.loc[titanic_test_data.Cabin.str[0] == 'E', 'Cabin'] = 5
titanic_test_data.loc[titanic_test_data.Cabin.str[0] == 'F', 'Cabin'] = 6
titanic_test_data.loc[titanic_test_data.Cabin.str[0] == 'G', 'Cabin'] = 7
# titanic_test_data.loc[titanic_test_data.Cabin.str[0] == 'T', 'Cabin'] = 8
titanic_test_data.Cabin = titanic_test_data.Cabin.astype('int32')

# Embarked
titanic_test_data.loc[titanic_test_data.Embarked == 'C', 'Embarked'] = 1
titanic_test_data.loc[titanic_test_data.Embarked == 'Q', 'Embarked'] = 2
titanic_test_data.loc[titanic_test_data.Embarked == 'S', 'Embarked'] = 3
titanic_test_data.Embarked.fillna(0, inplace=True)

titanic_test_data.Age.fillna(titanic_test_data.Age.mean(), inplace=True)
titanic_test_data.Fare.fillna(0, inplace=True)

titanic_test_data.isnull().sum()
titanic_test_data.head()
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
y = titanic_train_data.Survived
X = titanic_train_data[features]
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train.shape, x_test.shape, Y_train.shape, y_test.shape
titanic_test_data.head()
test_x = titanic_test_data[features]
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred1 = logreg.predict(x_test)
acc_log = round(logreg.score(x_test, y_test) * 100, 2)
acc_log
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, Y_pred1))
cm = pd.DataFrame(confusion_matrix(y_test, Y_pred1), ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
print(cm)
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train, Y_train)
Y_pred2 = svc.predict(x_test)
acc_svc = round(svc.score(x_test, y_test) * 100, 2)
acc_svc
print(classification_report(y_test, Y_pred2))
cm = pd.DataFrame(confusion_matrix(y_test, Y_pred2), ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
print(cm)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 
                           metric_params=None, n_jobs=1, n_neighbors=10, p=2, 
                           weights='uniform')
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(x_test)
acc_knn = round(knn.score(x_test, y_test) * 100, 2)
acc_knn
print(classification_report(y_test, knn_predictions))
cm = pd.DataFrame(confusion_matrix(y_test, knn_predictions), ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
print(cm)
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred7 = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_test, y_test) * 100, 2)
acc_decision_tree
print(classification_report(y_test, Y_pred7))
cm = pd.DataFrame(confusion_matrix(y_test, Y_pred7), ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
print(cm)
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest_predictions = random_forest.predict(x_test)
acc_random_forest = round(random_forest.score(x_test, y_test) * 100, 2)

acc_random_forest
print(classification_report(y_test, random_forest_predictions))
cm = pd.DataFrame(confusion_matrix(y_test, random_forest_predictions), ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
print(cm)
objects = ('Logistic Regression', 'SVC', 'KNN', 'Decision Tree', 'Random Forest')
x_pos = np.arange(len(objects))
accuracies1 = [acc_log, acc_svc, acc_knn, acc_decision_tree, acc_random_forest]
    
plt.bar(x_pos, accuracies1, align='center', alpha=0.5, color='b')
plt.xticks(x_pos, objects, rotation='vertical')
plt.ylabel('Accuracy')
plt.title('Classifier Outcome')
plt.show()
# Preparing data for Submission
test_random_forest_predictions = random_forest.predict(test_x)
test_Survived = pd.Series(test_random_forest_predictions, name="Survived")
Submission = pd.concat([titanic_test_data.PassengerId,test_Survived],axis=1)
Submission.to_csv('submission.csv', index=False)