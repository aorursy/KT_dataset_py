# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split









train_df= pd.read_csv("../input/train.csv")

test_df= pd.read_csv("../input/test.csv")
train_df.head()
train_df.corr()
train_df= train_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df= test_df.drop(['Name','Ticket'], axis=1)
train_df.head()
train_df['Embarked'].value_counts()
train_df["Embarked"] = train_df["Embarked"].fillna("S")

test_df['Embarked']= test_df['Embarked'].fillna('S')
embark_dummies_titanic  = pd.get_dummies(train_df['Embarked'])

train_df= train_df.join(embark_dummies_titanic)



embark_dummies_test= pd.get_dummies(test_df['Embarked'])

test_df= test_df.join(embark_dummies_test)
train_df.head()
sns.factorplot('Embarked','Survived', data=train_df,size=4,aspect=3)
train_df.corr().S
train_df.corr().C
train_df.corr().C
train_df['Sex'].isnull().sum()
train_df['Sex'].value_counts()
train_df.info()
def bar_chart(feature):

    survived = train_df[train_df['Survived']==1][feature].value_counts()

    dead = train_df[train_df['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(15,7))

    

bar_chart('Sex')

train_df.head()
test_df.info()
train_df= train_df.drop(['Cabin', 'Embarked', 'Sex'], axis=1)

test_df= test_df.drop(['Cabin','Embarked','Sex'], axis=1)
train_df.info()
sns.boxplot(train_df['Age'])
test_df.info()

train_df.corr().Age
average_age_titanic   = train_df["Age"].mean()

std_age_titanic       = train_df["Age"].std()

count_nan_age_titanic = train_df["Age"].isnull().sum()





average_age_test   = test_df["Age"].mean()

std_age_test       = test_df["Age"].std()

count_nan_age_test = test_df["Age"].isnull().sum()





rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)



rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)





train_df['Age'].dropna().astype(int)

test_df['Age'].dropna().astype(int)







train_df["Age"][np.isnan(train_df["Age"])] = rand_1

test_df["Age"][np.isnan(test_df["Age"])] = rand_2



train_df['Age'] = train_df['Age'].astype(int)

test_df['Age']    = test_df['Age'].astype(int)

        

train_df['Fare'] = train_df['Fare'].astype(int)





test_df['Fare']    = test_df['Fare'].fillna(test_df['Fare'].median())

        





        
train_df.info()
test_df.info()
X_train= train_df.drop('Survived', axis=1)

Y_train= train_df['Survived']

X_test= test_df.drop('PassengerId', axis=1).copy()
X_test.info()
train_df.head()
M_train, M_test, n_train, n_test= train_test_split(X_train, Y_train, test_size= 0.4, random_state= 42)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



c_space= np.logspace(-5, 8, 15)

param_grid= {'C': c_space, 'penalty': ['l1','l2']}



logreg= LogisticRegression(C= 0.0517, penalty= 'l2')



logreg.fit(M_train, n_train)

logreg.score(M_test, n_test)

logreg.fit(X_train, Y_train)

logreg.score(X_train, Y_train)

Y_pred= logreg.predict(X_test)

knn= KNeighborsClassifier(n_neighbors= 3)

knn.fit(M_train, n_train)





print(knn.score(M_test, n_test))
"""from sklearn.svm import SVC

clf= SVC(gamma= 'auto')

clf.fit(M_train, n_train)



clf.score(M_test, n_test)"""

"""from sklearn.ensemble import GradientBoostingClassifier as GBC

from sklearn.model_selection import GridSearchCV



gbr= GBC(max_depth=5, n_estimators= 100, random_state=1, subsample= 0.9)

gbr.fit(X_train, Y_train)

Y_pred= gbr.predict(X_test)"""

"""gsc= GridSearchCV(estimator= GBC(), 

                  param_grid= {'max_depth': range(4,7), 

                              'n_estimators': (150,100),

                               'subsample': (0.6,0.9)

                              }, cv=7, scoring= 'neg_mean_squared_error'

                 , verbose=0, n_jobs= -1)

grid_result= gsc.fit(X_train, Y_train)

best_params= grid_result.best_params_

print(best_params)"""
submission=pd.DataFrame({ 'PassengerId': test_df['PassengerId'], 'Survived': Y_pred

                        })



submission.to_csv('titanic.csv', index= False)