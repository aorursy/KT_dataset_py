# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk # models
import seaborn as sns# visualizations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data_train=pd.read_csv('../input/train.csv') #Read train data
data_test=pd.read_csv('../input/test.csv')#Read test data
print("SHAPE")
print("Training data: ", data_train.shape) #Examine shape of data
print("Testing data: ", data_test.shape)#Examine shape of data
print()

#Examine first 10 rows of data
data_train.head(10)

data_test.info()
#Check to see how many null values are in dataframe for each column.
print("NUMBER OF NULLS IN COLUMNS data_train: ")
data_train.isnull().sum()#Takes all null values and displays ammount for each coloumn
#Check to see how mnay null values are in dataframe for each column.
print("NUMBER OF NULLS IN COLUMNS data_test: ")
data_test.isnull().sum()
# Create CabinBool feature
data_train["CabinBool"] = (data_train["Cabin"].notnull().astype('int'))
data_test["CabinBool"] = (data_test["Cabin"].notnull().astype('int'))
sns.lmplot(x="PassengerId", y="Fare", data=data_train, fit_reg=True)
data_train.loc[data_train['Fare'] > 300] #Show all passengers that paid more than 300
data_train = data_train[data_train.Fare < 300]
sns.lmplot(x="PassengerId", y="Fare", data=data_train, fit_reg=True)
data_train.drop('Cabin', axis = 1, inplace = True)
data_test.drop('Cabin', axis = 1, inplace = True)

data_train.head() # Check to see if the replacement worked...
#Calculate correlations
corr=data_train.corr()

#Heatmap
sns.heatmap(corr, cmap="Blues")
data_train["Age"].fillna(data_train.groupby("Sex")["Age"].transform("mean"), inplace=True)
data_test['Age'].fillna(data_test.groupby('Sex')['Age'].transform("mean"), inplace=True)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
data_train['AgeGroup'] = pd.cut(data_train["Age"], bins, labels = labels)
data_test['AgeGroup'] = pd.cut(data_test["Age"], bins, labels = labels)

# Map each age value into a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
data_train['AgeGroup'] = data_train['AgeGroup'].map(age_mapping)
data_test['AgeGroup'] = data_test['AgeGroup'].map(age_mapping)


# Drop Age column from each dataset now that new column 'FareGroups' has been made.
data_train = data_train.drop(['Age'], axis = 1)
data_test = data_test.drop(['Age'], axis = 1)

data_train.loc[data_train.Embarked.isnull()]
data_train['Embarked'].fillna("S", inplace = True)
data_test['Fare'].fillna(data_test['Fare'].mean(), inplace = True)
                                                            
#Check to see how many null values are in dataframe for each column.
print("NUMBER OF NULLS IN COLUMNS: ")
data_train.isnull().sum()

# Split Fare column in each dataset into four different labels.
data_train['FareGroups'] = pd.qcut(data_train['Fare'], 4, labels = [1, 2, 3, 4])
data_test['FareGroups'] = pd.qcut(data_test['Fare'], 4, labels = [1, 2, 3, 4])


# Drop Fare column from each dataset now that new column 'FareGroups' has been made.
data_train = data_train.drop(['Fare'], axis = 1)
data_test = data_test.drop(['Fare'], axis = 1)

data_train = pd.get_dummies(data_train, columns=['Sex', 'Embarked'], drop_first=True)
data_test = pd.get_dummies(data_test, columns=['Sex', 'Embarked'], drop_first=True)

data_train = data_train.drop(["PassengerId","Name","Ticket"], axis=1)
data_test = data_test.drop(['Name','Ticket'], axis=1)
data_test.tail()
data_train.head()
data_test.tail()
X_train= data_train.drop(["Survived"], axis=1)
Y_train= data_train.Survived
X_test= data_test.drop(['PassengerId'], axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_sub = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, Y_train)
Y_pred = naive_bayes.predict(X_test)
acc_naive_bayes = round(naive_bayes.score(X_train, Y_train) * 100, 2)
acc_naive_bayes

#from xgboost import XGBClassifier

#xgb = XGBClassifier(n_estimators=200)
#xgb.fit(X_train, Y_train)
#Y_pred = xgb.predict(X_test)
#acc_xgb = round(xgb.score(X_train, Y_train)*100, 2)
#acc_xgb
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Decision Tree'], 
    
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_naive_bayes, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({"PassengerId": data_test["PassengerId"],
                           "Survived": Y_pred_sub
                          })
submission.to_csv('submit.csv', index=False)