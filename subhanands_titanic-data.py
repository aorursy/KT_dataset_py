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

import seaborn as sns

import numpy as np

%matplotlib inline

sns.set_style('whitegrid')



train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
train_data.head()
train_data.info()
train_data.describe()
sns.heatmap(train_data.isnull())
sns.countplot(x='Survived', data=train_data)
f, axes = plt.subplots(1, 2, figsize=(15,7))

sns.countplot(x='Survived',hue='Sex', data=train_data, color="b", ax=axes[0])

sns.countplot(x='Survived',hue='Pclass', data=train_data, color="r", ax=axes[1])
sns.distplot(train_data['Age'].dropna())
train_data['Fare'].hist(bins=30)

train_data['SibSp'].hist(bins=30)
train_data['Parch'].hist(bins=30)
sns.pairplot(train_data)
train_data['Age']= train_data[['Age']].fillna(value=train_data['Age'].mean())
sns.heatmap(train_data.isnull())
train_data= train_data.drop(['Cabin','PassengerId','Name','Ticket'], axis=1)
sex=pd.get_dummies(train_data['Sex'], drop_first=True)

embarked= pd.get_dummies(train_data['Embarked'], drop_first=True)
train_data.drop(['Sex','Embarked'], axis=1,inplace=True)
train_data=pd.concat([train_data,sex,embarked], axis=1)
train_data.head()
from sklearn.model_selection import train_test_split

X=train_data.drop(['Survived'], axis=1)

y=train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30,random_state=501)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,y_train)
predict_with_lr=lr.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

report_lr=classification_report(y_test,predict_with_lr,output_dict=True)

print ("Logistic Regression model \n", classification_report(y_test,predict_with_lr))

print('\n')

print ("Logistic Regression model \n",confusion_matrix(y_test,predict_with_lr))

acc_matrix=pd.DataFrame({'Model' : ['Logistic Regression'],

                        'Accuracy':[report_lr['accuracy']]})

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

predict_with_knn=knn.predict(X_test)
report_knn=classification_report(y_test,predict_with_knn, output_dict=True)

dict={'Model' : 'K Nearest Neighbors',

                        'Accuracy':report_knn['accuracy']}

acc_matrix=acc_matrix.append(dict, ignore_index=True)

print ("K Nearest Neighbors \n", classification_report(y_test,predict_with_knn))

print('\n')

print ("K Nearest Neighbors \n",confusion_matrix(y_test,predict_with_knn))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=800)

rf.fit(X_train,y_train)

predict_with_rf=rf.predict(X_test)
report_rf=classification_report(y_test,predict_with_rf, output_dict=True)

dict={'Model' : 'Random Forest',

                        'Accuracy':report_rf['accuracy']}

acc_matrix=acc_matrix.append(dict, ignore_index=True)

print ("Random Forest \n", classification_report(y_test,predict_with_rf))

print('\n')

print ("Random Forest \n",confusion_matrix(y_test,predict_with_rf))
from sklearn.svm import SVC

svc=SVC()

svc.fit(X_train,y_train)

predict_with_svc=svc.predict(X_test)
report_svm=classification_report(y_test,predict_with_svc, output_dict=True)

dict={'Model' : 'SVM Classifier',

                        'Accuracy':report_svm['accuracy']}

acc_matrix=acc_matrix.append(dict, ignore_index=True)

print ("SVM Classification \n", classification_report(y_test,predict_with_svc))

print('\n')

print ("SVM Classification \n",confusion_matrix(y_test,predict_with_svc))
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
predict_with_svc_grid=grid.predict(X_test)

report_svm_gs=classification_report(y_test,predict_with_svc_grid, output_dict=True)

dict={'Model' : 'SVM Classifier after Grid search',

                        'Accuracy':report_svm_gs['accuracy']}

acc_matrix=acc_matrix.append(dict, ignore_index=True)

print ("SVM Classification with Grid \n", classification_report(y_test,predict_with_svc_grid))

print('\n')

print ("SVM Classification with Grid\n",confusion_matrix(y_test,predict_with_svc_grid))
acc_matrix
test_data.head()
sns.heatmap(test_data.isnull())
filter_test_data=test_data.drop(['Cabin','PassengerId','Name','Ticket'], axis=1)
filter_test_data['Age']=filter_test_data[['Age']].fillna(value=filter_test_data['Age'].mean())

sex_test=pd.get_dummies(filter_test_data['Sex'], drop_first=True)

embarked_test= pd.get_dummies(filter_test_data['Embarked'], drop_first=True)

filter_test_data.drop(['Sex','Embarked'], axis=1,inplace=True)

filter_test_data=pd.concat([filter_test_data,sex_test,embarked_test], axis=1)
filter_test_data.isnull().sum()
filter_test_data['Fare']=filter_test_data[['Fare']].fillna(value=filter_test_data['Age'].mean())
final_pred=rf.predict(filter_test_data)
submission= pd.DataFrame({ 

    'PassengerId': test_data['PassengerId'],

    'Survived': final_pred })

submission.to_csv("Submission.csv", index=False)