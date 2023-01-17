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
import matplotlib.pyplot as plt

import seaborn as sns 

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

submission=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

train.head()
print(train.shape)

print('-'*50)

print(test.shape)
train.info()

print('-'*50)

test.info()

train.isna().sum()

test.isna().sum()
train.describe()
train.describe(include=['O'])
train.head()
train['Pclass'].value_counts()
pd.pivot_table(data=train,index='Pclass',values='Survived',aggfunc=np.mean)
g=sns.FacetGrid(train,col='Survived',row='Pclass')

g.map(plt.hist,'Age',bins=20)
train['Sex'].value_counts()
pd.pivot_table(data=train,index='Sex',values='Survived',aggfunc=np.mean)
g=sns.FacetGrid(train,col='Survived')

g.map(sns.countplot,'Sex')
pd.pivot_table(index=['Embarked','Sex','Pclass'],data=train,values='Survived',aggfunc=np.mean)
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
train=train.drop(columns=['Ticket','Cabin'])

test=test.drop(columns=['Ticket','Cabin'])
combine=[train,test]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

pd.pivot_table(index='Title',values='Survived',data=train,aggfunc=np.mean).sort_values(by='Survived',ascending=False)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train.head()
train.drop(columns=['Name','PassengerId'],inplace=True)
test.drop('Name',inplace=True,axis=1)
combine=[train,test]
sex_dict= {'female': 1, 'male': 0}

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map(sex_dict ).astype(int)
train
guess_ages = np.zeros((2,3))

guess_ages

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

combine = [train, test]
for dataset in combine:

    dataset['Family']=dataset['SibSp']+dataset['Parch']+1

    
pd.pivot_table(index='Family',data = train , values='Survived',aggfunc=np.mean).sort_values(by='Family',ascending=True)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1





train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train = train.drop(['Parch', 'SibSp', 'Family'], axis=1)

test = test.drop(['Parch', 'SibSp', 'Family'], axis=1)

combine = [train, test]
train.head()
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(train.Embarked.dropna().mode()[0])
train['Embarked'].value_counts()
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train.head()
test.isna().sum()
test['Fare']=test['Fare'].dropna()
test[test['Fare'].isna()]
test.drop(152,axis=0,inplace=True)
test.isna().sum()
train 
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

combine = [train, test]

    

train.head()
test.head()
X=train.drop('Survived',axis=1)

y=train['Survived']

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=20)
# Import necessary modules

from scipy.stats import randint



from sklearn.model_selection import RandomizedSearchCV



# Setup the parameters and distributions to sample from: param_dist

param_dist = {"max_depth": [3, None],

              "max_features": randint(1, 9),

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"]}



# Instantiate a Decision Tree classifier: tree

tree = DecisionTreeClassifier()



# Instantiate the RandomizedSearchCV object: tree_cv

tree_cv = RandomizedSearchCV(tree,param_dist, cv=5)



# Fit it to the data

tree_cv.fit(X_train,y_train)



# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))

print("Best score is {}".format(tree_cv.best_score_))



acc_decision_tree=tree_cv.best_score_
param_dist = {'n_estimators':np.arange(1,200)}



# Instantiate a Decision Tree classifier: tree

random_forest = RandomForestClassifier()



# Instantiate the RandomizedSearchCV object: tree_cv

random_forest_cv = RandomizedSearchCV(random_forest,param_dist, cv=5)



# Fit it to the data

random_forest_cv.fit(X_train,y_train)



# Print the tuned parameters and score

print("Tuned Random Forest Parameters: {}".format(random_forest_cv.best_params_))

print("Best score is {}".format(random_forest_cv.best_score_))

acc_random_forest=random_forest_cv.best_score_
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_test, y_test) , 2)

acc_linear_svc
knn = KNeighborsClassifier(n_neighbors = 11)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_test, y_test) , 2)

acc_knn
# Logistic regression

from sklearn.pipeline import Pipeline



from sklearn.preprocessing import StandardScaler



logregpipe = Pipeline([('scale', StandardScaler()),

                   ('logreg',LogisticRegression(multi_class="multinomial",solver="lbfgs"))])



# Gridsearch to determine the value of C

param_grid = {'logreg__C':np.arange(0.01,100,10)}

logreg_cv = GridSearchCV(logregpipe,param_grid,cv=5,return_train_score=True)

logreg_cv.fit(X_train,y_train)

print(logreg_cv.best_params_)



bestlogreg = logreg_cv.best_estimator_

bestlogreg.fit(X_train,y_train)

bestlogreg.coef_ = bestlogreg.named_steps['logreg'].coef_

acc_log=bestlogreg.score(X_test,y_test)
svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_test,y_test) , 2)

acc_svc
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest' ,'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest ,acc_linear_svc,  acc_decision_tree]})

models.sort_values(by='Score', ascending=False)