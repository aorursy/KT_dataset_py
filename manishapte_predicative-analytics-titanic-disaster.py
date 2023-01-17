# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



### For data analysis and wrangling

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



# For visualization

import seaborn as sns

import matplotlib.pyplot as plt

#%matplotlib.inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC , LinearSVC

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the train and Display the First 5 Rows of Titanic Dataset

train = pd.read_csv("../input/titanic-solution-for-beginners-guide/train.csv")

train.head()
# Load the test and Display the First 5 Rows of Titanic Dataset

test = pd.read_csv("../input/titanic-solution-for-beginners-guide/test.csv")

test.head()
combine = [train,test]
print('Number of Rows and column in training dataset',train.shape)

print('Number of Rows and column in testing dataset',test.shape)
print('Training columns : ')

print(train.columns.values)

print('Testing columns : ')

print(test.columns.values)
target_variable = list(set(train) - set(test))

print('Target Column is :',target_variable)
train.info()
# select_dtypes : Return a subset of the DataFrame’s columns based on the column dtypes.



#it will return column which contain discrete (integer) value  

discrete_data = train.select_dtypes(include=['int64'])

#it will return column which contain continous (float) value

continous_data = train.select_dtypes(include=['float64'])

#it will return column which contain categorical (object) value

categorical_data = train.select_dtypes(include=['object'])
print('Discrete Features : ',discrete_data.columns.values)

print('Continous Features : ',continous_data.columns.values)

print('Categorical Features : ',categorical_data.columns.values)
categorical_data.head()
discrete_data.tail()
continous_data.head()
train.head()
train.tail()
sns.heatmap(train.isnull(),yticklabels=False,cmap='bwr')
train.info()
sns.heatmap(test.isnull(),yticklabels=False,cmap='bwr')
train.info()

print('--'*40)

test.info()
train.describe()
categorical_data.describe()
discrete_data.describe()
continous_data.describe()
from numpy import percentile

from numpy.random import rand



print('Sibling Ratio : ',percentile(train['SibSp'],[10, 20, 30, 40, 50, 60,65,68, 70, 80, 90]))

print('Fare Charge : ',percentile(train['Fare'],[10, 20, 30, 40, 50, 60,70, 80, 90,95,99]))
train['Pclass'].value_counts()
train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['Sex','Survived']].groupby('Sex',as_index=False).mean().sort_values(by='Survived',ascending=False)
grid1 =sns.FacetGrid(train,col='Survived')

grid1.map(plt.hist, 'Age', bins=20)
grid2 = sns.FacetGrid(train,col='Survived',row='Pclass')

grid2.map(plt.hist,'Age',alpha=0.5,bins=20)

grid2.add_legend()
grid2 = sns.FacetGrid(train,row='Embarked' , size=5.2, aspect=1.6)

grid2.map(sns.pointplot,'Pclass','Survived','Sex', palette='deep')

grid2.add_legend()

grid2 = sns.FacetGrid(train,row='Embarked',col='Survived',size=3.0,aspect=2.0)

grid2.map(sns.barplot,'Sex','Fare',alpha=0.5,ci=None)

grid2.add_legend()
print('Before : ',train.shape,test.shape)

columns = ['Cabin','Ticket']

new_train = train.drop(columns,axis=1)

new_test = test.drop(columns,axis=1) 

combine = [new_train,new_test]

print('After : ',new_train.shape,new_test.shape)
new_train.head()
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

dataset['Title'].head()
pd.crosstab(new_train['Title'],new_train['Sex'])
new_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Major','Rev','Sir','Dona','Lady'],'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

new_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived',ascending=False)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



new_train.head()
new_train = new_train.drop(['Name', 'PassengerId'], axis=1)

new_test = new_test.drop(['Name'], axis=1)

combine = [new_train, new_test]

new_train.shape, new_test.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({"male" : 0 , "female" : 1}).astype(int)



new_train.head()
new_train.info()
grid = sns.FacetGrid(new_train, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages



for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess =  guess_df.median()

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

new_train.head()
new_train.info()
new_train['AgeBand'] = pd.cut(new_train['Age'], 5)

new_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
new_train.head()
for dataset in combine:

    dataset.loc[dataset['Age'] <= 16,'Age'] = 0

    dataset.loc[(dataset['Age']>16) & (dataset['Age'] <=32),'Age'] = 1

    dataset.loc[(dataset['Age']>32) & (dataset['Age'] <=48), 'Age'] = 2

    dataset.loc[(dataset['Age']>38) & (dataset['Age'] <=64),'Age'] = 3

    dataset.loc[(dataset['Age']>64), 'Age'] = 4
new_train.head()
new_test.head()
for dataset in combine:

    dataset['Familysize'] = dataset['Parch'] + dataset['SibSp'] + 1

new_train[['Familysize', 'Survived']].groupby(['Familysize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['Familysize'] == 1,'IsAlone'] = 1
new_train[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean().sort_values(by='Survived',ascending=False)
new_train = new_train.drop(['Parch', 'SibSp', 'Familysize'], axis=1)

new_test = new_test.drop(['Parch', 'SibSp', 'Familysize'], axis=1)

combine = [new_train, new_test]



new_train.head()
for dataset in combine:

    dataset['Age*class'] = dataset.Age * dataset.Pclass

new_train.loc[:,['Age*class','Age','Pclass']].head()
freq_port = new_train.Embarked.dropna().mode()[0]

freq_port
dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

new_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    print(dataset.info())

    print(dataset.dropna(how='any',inplace=True))

    print(dataset.info())
from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder()  


for dataset in combine:

    dataset['Embarked'] = label_encoder.fit_transform(dataset['Embarked'])
dataset['Embarked'].unique() 

new_train.head()
new_test['Fare'].fillna(new_test['Fare'].dropna().median(), inplace=True)

new_test.head()
new_train['FareBand'] = pd.qcut(new_train['Fare'], 4,duplicates='drop')

new_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



new_train = new_train.drop(['FareBand','AgeBand'], axis=1)

combine = [new_train, new_test]

    

new_train.head(10)
X_train = new_train.drop("Survived", axis=1)

Y_train = new_train["Survived"]

X_test  = new_test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()

logistic_model.fit(X_train,Y_train)

### Check the training accuracy

training_accuracy = logistic_model.score(X_train,Y_train)

y_pred = logistic_model.predict(X_test)

acc_log = round((training_accuracy)*100,2)

print('Training Accuracy For the Logistic Regression Model is ',acc_log)
### Coeffecient of each feature

coeff_df = pd.DataFrame(new_train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logistic_model.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
from sklearn.svm import SVC

svc_model = SVC()

svc_model.fit(X_train,Y_train)

acc_svc = round(svc_model.score(X_train,Y_train) * 100 ,2)

print('Training Accuracy For the SVC Model ',acc_svc)

y_pred = svc_model.predict(X_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print('Training Accuracy For the KNN Neighbour is ',acc_knn)
from sklearn.naive_bayes import GaussianNB

naive_bayes_model = GaussianNB()

naive_bayes_model.fit(X_train,Y_train)

acc_gaussian = round(naive_bayes_model.score(X_train,Y_train)*100,2)

print('Training Accuracy For the Navie Bayes Model is : ',acc_gaussian)

y_pred = naive_bayes_model.predict(X_test)
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print('Training Accuracy For the Decision Tree Classifier Model is :',acc_decision_tree)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print('Training Accuracy For the RandomForestClassifier Model is :',acc_random_forest)
import xgboost



xgb = xgboost.XGBClassifier(colsample_bytree=0.8, subsample=0.5,

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.8, n_estimators=2000,

                             reg_alpha=0.1, reg_lambda=0.3, gamma=0.01, 

                             silent=1, random_state =7, nthread = -1)



xgb.fit(X_train, Y_train)

Y_pred = xgb.predict(X_test)

acc_xgb = round(xgb.score(X_train, Y_train) * 100, 2)

print('Training Accuracy For the XGBoost Model is :',acc_xgb)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

               'Naive Bayes',  

              'Decision Tree','Random Forest','XGBoost'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_gaussian, acc_decision_tree,acc_random_forest,acc_xgb]})

models.sort_values(by='Score', ascending=False)