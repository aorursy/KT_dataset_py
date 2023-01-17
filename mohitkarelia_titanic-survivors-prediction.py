# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

sb.set_style('whitegrid')

sb.set_palette('pastel')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/titanic/train.csv")

supplement = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")



test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

pid=test_data.PassengerId
data.head()
data.dtypes
test_data.dtypes
supplement.dtypes
#Finding Null Values

print("Null Values in Training data: ")

print(data.isnull().sum())

print("")

print("Null Values in Test Data: ")

print(test_data.isnull().sum())
print("Training Data")

print("Mean:",data['Age'].mean())

print("Median",data['Age'].median())

print("Mode",data['Age'].mode())



print("")

print("Test data")

print("Mean:",test_data['Age'].mean())

print("Median",test_data['Age'].median())

print("Mode",test_data['Age'].mode())

data['Age'] = data['Age'].fillna(data['Age'].median())

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])



test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
columns=['PassengerId','Cabin','Ticket','Name']

data=data.drop(columns,axis=1)

test_data=test_data.drop(columns,axis=1)
data.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Sex'] = le.fit_transform(data['Sex'])

test_data['Sex'] = le.fit_transform(test_data['Sex'])

data['Embarked'] = le.fit_transform(data['Embarked'])

test_data['Embarked'] = le.fit_transform(test_data['Embarked'])

data['Age'] = data['Age'].astype(int)
plt.figure(figsize = (12,6))

sb.heatmap(data.corr(),annot=True)
#Percentage Survived for each category

target=['Survived']

selected=['Sex','Pclass','Embarked','SibSp','Parch']

for x in selected:

    print('Survival Percentage By',x)

    print(data[[x, target[0]]].groupby(x,as_index=False).mean(),'\n')

# Sex - 0: Female, 1: Male

# Embarked - 0: C, 1: Q, 2: S

plt.figure(figsize = (15,6))

plt.hist(x = [data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']],stacked=True,color = ['green','red'],label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Survived')

plt.title('Survival By Age')

plt.legend()

plt.figure(figsize = (15,6))

plt.hist(x = [data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']],stacked=True,color = ['green','red'],label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Survived')

plt.title('Survival By Fare')

plt.legend()
plt.figure(figsize = (15,6))

plt.hist(x = [data[data['Survived']==1]['Pclass'],data[data['Survived']==0]['Pclass']],stacked=True,color = ['green','red'],label = ['Survived','Dead'])

plt.xlabel('PClass')

plt.ylabel('Survived')

plt.title('Survival By Class')

plt.legend()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score

y = data['Survived']

X = data[['Pclass','Sex','Fare','Embarked']]

test_data.drop(['Age','SibSp','Parch'],1,inplace=True)
test_data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
param_grid = {'n_neighbors':np.arange(1,6)}

grid_knn = GridSearchCV(KNeighborsClassifier(),param_grid,cv=3)

grid_knn.fit(X,y)
grid_knn.best_params_
clf1 = grid_knn.best_estimator_
clf1.fit(X_train,y_train)

y_train_pred = cross_val_predict(clf1,X_train,y_train,cv=3)

print("Confusion Matrix:",confusion_matrix(y_train,y_train_pred))

print("")

print("Precision Score: ",precision_score(y_train,y_train_pred))

print("")

print("Recall Score: ",recall_score(y_train,y_train_pred))

print("")

print("Cross Val Score in Sample",cross_val_score(clf1,X_train,y_train,cv=3,scoring='accuracy').mean())

print("")

print("Cross Val Score out Sample",cross_val_score(clf1,X_test,y_test,cv=3,scoring='accuracy').mean())
clf2 = LogisticRegression(C = 1,max_iter = 500)

clf2.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf2,X_train,y_train,cv=3)

print("Confusion Matrix:",confusion_matrix(y_train,y_train_pred))

print("")

print("Precision Score: ",precision_score(y_train,y_train_pred))

print("")

print("Recall Score: ",recall_score(y_train,y_train_pred))

print("")

print("Cross Val Score in Sample",cross_val_score(clf2,X_train,y_train,cv=3,scoring='accuracy').mean())

print("")

print("Cross Val Score out Sample",cross_val_score(clf2,X_test,y_test,cv=3,scoring='accuracy').mean())
clf3 = SVC(kernel='poly',C = 1000)

clf3.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf3,X_train,y_train,cv=3)

print("accuracy_score",accuracy_score(y_train,y_train_pred))

print("Confusion Matrix:",confusion_matrix(y_train,y_train_pred))

print("")

print("Precision Score: ",precision_score(y_train,y_train_pred))

print("")

print("Recall Score: ",recall_score(y_train,y_train_pred))

print("")

print("Cross Val Score in Sample",cross_val_score(clf3,X_train,y_train,cv=3,scoring='accuracy').mean())

print("")

print("Cross Val Score out Sample",cross_val_score(clf3,X_test,y_test,cv=3,scoring='accuracy').mean())
param_grid = {'max_depth':np.arange(1,4),'min_samples_leaf':np.arange(1,3)}

grid_tree = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=6)

grid_tree.fit(X,y)
grid_tree.best_params_
#clf4 = grid_tree.best_estimator_

#clf4.fit(X_train,y_train)

clf4 = DecisionTreeClassifier(max_depth=3,min_samples_leaf=1)
from sklearn.metrics import accuracy_score

y_train_pred = cross_val_predict(clf4,X_train,y_train,cv=3)

print("accuracy score:",accuracy_score(y_train,y_train_pred))

print("Confusion Matrix:",confusion_matrix(y_train,y_train_pred))

print("")

print("Precision Score: ",precision_score(y_train,y_train_pred))

print("")

print("Recall Score: ",recall_score(y_train,y_train_pred))

print("")

print("Cross Val Score in Sample",cross_val_score(clf4,X_train,y_train,cv=3,scoring='accuracy').mean())

print("")

print("Cross Val Score out Sample",cross_val_score(clf4,X_test,y_test,cv=3,scoring='accuracy').mean())
clf5 = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=4, max_features='auto',

                       max_leaf_nodes=2, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=2, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=5,

                       random_state=1, verbose=0,

                       warm_start=False)

clf5.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf5,X_train,y_train,cv=5)

from sklearn.metrics import accuracy_score

print("accuracy score:",accuracy_score(y_train,y_train_pred))

print("Confusion Matrix:",confusion_matrix(y_train,y_train_pred))

print("")

print("Precision Score: ",precision_score(y_train,y_train_pred))

print("")

print("Recall Score: ",recall_score(y_train,y_train_pred))

print("")

print("Cross Val Score in Sample",cross_val_score(clf5,X_train,y_train,cv=3,scoring='accuracy').mean())

print("")

print("Cross Val Score out Sample",cross_val_score(clf5,X_test,y_test,cv=3,scoring='accuracy').mean())
clf6 = BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1),n_estimators = 1000,bootstrap=True,n_jobs=-1)

clf6.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf6,X_train,y_train,cv=3)

print("accuracy_score",accuracy_score(y_train,y_train_pred))

print("Confusion Matrix:",confusion_matrix(y_train,y_train_pred))

print("")

print("Precision Score: ",precision_score(y_train,y_train_pred))

print("")

print("Recall Score: ",recall_score(y_train,y_train_pred))

print("")

print("Cross Val Score in Sample",cross_val_score(clf6,X_train,y_train,cv=3,scoring='accuracy').mean())

print("")

print("Cross Val Score out Sample",cross_val_score(clf6,X_test,y_test,cv=3,scoring='accuracy').mean())
clf7 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2),n_estimators=1000)

clf7.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf7,X_train,y_train,cv=3)

print("accuracy_score",accuracy_score(y_train,y_train_pred))

print("Confusion Matrix:",confusion_matrix(y_train,y_train_pred))

print("")

print("Precision Score: ",precision_score(y_train,y_train_pred))

print("")

print("Recall Score: ",recall_score(y_train,y_train_pred))

print("")

print("Cross Val Score in Sample",cross_val_score(clf7,X_train,y_train,cv=3,scoring='accuracy').mean())

print("")

print("Cross Val Score out Sample",cross_val_score(clf7,X_test,y_test,cv=3,scoring='accuracy').mean())
clf = DecisionTreeClassifier(max_depth=3,min_samples_leaf=1)

clf.fit(X,y)
predicted_values = clf.predict(test_data)

predicted_values
output = pd.DataFrame({'PassengerId': pid, 'Survived': predicted_values})

output.to_csv('my_submission21_.csv', index=False)
