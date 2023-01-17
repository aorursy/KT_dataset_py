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
#

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.info()
test_data.info()
train_data.head()
test_data.head()
train_data.describe()
women  = len(train_data.loc[train_data["Sex"]== "female"])
men = len(train_data.loc[train_data["Sex"]== "male"])
print("number of women:",women)
print("number of men:" , men)
print(train_data.columns)
print(len(train_data.columns))
sns.heatmap(train_data.corr(),annot= True)
full_data = [train_data, test_data]
sex_dict = {"female":0, "male":1}
for data_df in full_data:
    data_df['Sex'] = data_df['Sex'].apply(lambda x:sex_dict[x])

for data_df in full_data:
    data_df['Title']  = data_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data_df['Title'] = data_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data_df['Title'] = data_df['Title'].replace('Mlle', 'Miss')
    data_df['Title'] = data_df['Title'].replace('Ms', 'Miss')
    data_df['Title'] = data_df['Title'].replace('Mme', 'Mrs')

    
title_dict = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for data_df in full_data: 
    data_df['Title'] = data_df['Title'].apply(lambda x: title_dict[x])
    data_df['Title'] = data_df['Title'].fillna(0)
    


def AgeGroup(age):
    if(age <= 16):
        return 0 
    elif age > 16 and age <= 32:
        return 1
    elif age>32 and age <=48:
        return 2 
    elif age>48 and age <= 64:
        return 3
    else:
        return 4
    
for data_df in full_data:
    age_avg = data_df['Age'].mean()
    age_std = data_df['Age'].std()
    age_null_count = data_df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    data_df['Age'][np.isnan(data_df['Age'])] = age_null_random_list
    data_df['Age'] = data_df['Age'].astype(int)
    data_df['AgeGoup'] = data_df['Age'].apply(AgeGroup)

def Alone(familysize):
    if familysize ==1:
        return 1 
    else:
        return 0

for data_df in full_data:
    data_df['Family_size'] = data_df['SibSp'] + data_df['Parch'] + 1
    data_df['IsAlone'] = data_df['Family_size'].apply(Alone)
embarked_dict= {'S': 0, 'C': 1, 'Q': 2}
for data_df in full_data:
    data_df['Embarked'] = data_df['Embarked'].fillna('S')
    data_df['Embarked'] = data_df['Embarked'].apply(lambda x: embarked_dict[x])
#type(train_data['Cabin'].iloc[1])
def HasCabin(cabin):
    if type(cabin) == str:
        return 1
    else:
        return 0
    
for data_df in full_data:
    data_df['HasCabin'] = data_df['Cabin'].apply(HasCabin)
def FareGroup(fare):
    if fare <= 7.91:
        return 0;
    elif fare >7.91 and fare <=14.454:
        return 1
    elif fare >14.454 and fare <=31:
        return 2
    else:
        return 3

for data_df in full_data:
    data_df['Fare'] = data_df['Fare'].fillna(data_df['Fare'].median())
    data_df['FareGroup'] = data_df['Fare'].apply(FareGroup)
feature_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
for data_df in full_data:
    data_df.drop(columns = feature_columns, inplace = True)

    
#train_data.info()
train_data.head(10)
# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
x_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]
x_test  = test_data.copy()
x_train.shape,y_train.shape
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
acc_gaussian
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
acc_perceptron

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)
acc_linear_svc
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
acc_sgd
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
