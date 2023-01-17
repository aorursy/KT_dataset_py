# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

sns.set()
raw_data_train=pd.read_csv('/kaggle/input/titanic/train.csv')

raw_data_test=pd.read_csv('/kaggle/input/titanic/test.csv')
raw_data_train.head()
raw_data_test.head()
raw_data_train.info()
raw_data_test.info()
data_train=raw_data_train[['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']]

data_train.head()
data_test=raw_data_test[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']]

data_test.head()
data_train['Sex']=data_train['Sex'].map({'male':1,'female':0})

data_test['Sex']=data_test['Sex'].map({'male':1,'female':0})
data_test.head()
data_train.head()
data_train.info()
data_train['Age']=data_train['Age'] = data_train['Age'].fillna(data_train['Age'].mean())
data_train.info()
data_test['Age']=data_test['Age'] = data_test['Age'].fillna(data_test['Age'].mean())

data_test.info()
data_test['Fare']=data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].mean())

data_test.info()
x_train=data_train.iloc[:,:-1].values

y_train=data_train.iloc[:,-1].values

x_test=data_test.iloc[:,1:7].values
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(x_train, y_train)

y_pred_log_reg = clf.predict(x_test)

acc_log_reg = round( clf.score(x_train, y_train) * 100, 2)

print (str(acc_log_reg) + ' percent')
from sklearn.svm import SVC

clf = SVC()

clf.fit(x_train, y_train)

y_pred_svc = clf.predict(x_test)

acc_svc = round(clf.score(x_train, y_train) * 100, 2)

print (acc_svc)
from sklearn.svm import LinearSVC

clf = LinearSVC()

clf.fit(x_train, y_train)

y_pred_linear_svc = clf.predict(x_test)

acc_linear_svc = round(clf.score(x_train, y_train) * 100, 2)

print (acc_linear_svc)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 3)

clf.fit(x_train, y_train)

y_pred_knn = clf.predict(x_test)

acc_knn = round(clf.score(x_train, y_train) * 100, 2)

print (acc_knn)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(x_train, y_train)

y_pred_decision_tree = clf.predict(x_test)

acc_decision_tree = round(clf.score(x_train, y_train) * 100, 2)

print (acc_decision_tree)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)

clf.fit(x_train, y_train)

y_pred_random_forest = clf.predict(x_test)

acc_random_forest = round(clf.score(x_train, y_train) * 100, 2)

print (acc_random_forest)
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(x_train, y_train)

y_pred_gnb = clf.predict(x_test)

acc_gnb = round(clf.score(x_train, y_train) * 100, 2)

print (acc_gnb)
from sklearn.linear_model import Perceptron

clf = Perceptron(max_iter=5, tol=None)

clf.fit(x_train, y_train)

y_pred_perceptron = clf.predict(x_test)

acc_perceptron = round(clf.score(x_train, y_train) * 100, 2)

print (acc_perceptron)
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(max_iter=5, tol=None)

clf.fit(x_train, y_train)

y_pred_sgd = clf.predict(x_test)

acc_sgd = round(clf.score(x_train, y_train) * 100, 2)

print (acc_sgd)
from sklearn.metrics import confusion_matrix

import itertools



clf = RandomForestClassifier(n_estimators=100)

clf.fit(x_train, y_train)

y_pred_random_forest_training_set = clf.predict(x_train)

acc_random_forest = round(clf.score(x_train, y_train) * 100, 2)

print ("Accuracy: %i %% \n"%acc_random_forest)



class_names = ['Survived', 'Not Survived']



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)

np.set_printoptions(precision=2)



print ('Confusion Matrix in Numbers')

print (cnf_matrix)

print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]



print ('Confusion Matrix in Percentage')

print (cnf_matrix_percent)

print ('')



true_class_names = ['True Survived', 'True Not Survived']

predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']



df_cnf_matrix = pd.DataFrame(cnf_matrix, 

                             index = true_class_names,

                             columns = predicted_class_names)



df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 

                                     index = true_class_names,

                                     columns = predicted_class_names)



plt.figure(figsize = (15,5))



plt.subplot(121)

sns.heatmap(df_cnf_matrix, annot=True, fmt='d')



plt.subplot(122)

sns.heatmap(df_cnf_matrix_percent, annot=True)


models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 

              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 

              'Perceptron', 'Stochastic Gradient Decent'],

    

    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 

              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 

              acc_perceptron, acc_sgd]

    })



models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": data_test["PassengerId"],

        "Survived": y_pred_random_forest

    })



submission.to_csv('submission.csv', index=False)