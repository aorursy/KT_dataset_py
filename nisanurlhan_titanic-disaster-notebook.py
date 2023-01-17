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
train_df=pd.read_csv("../input/titanic/train.csv")

test_df=pd.read_csv("../input/titanic/train.csv")

gender_sub=pd.read_csv("../input/titanic/train.csv")

df=[train_df,test_df]
train_df
train_df.info()
test_df.info()
train_df.head()
train_df.describe()
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived')
train_df[['Age','Survived']].groupby(['Age'],as_index=False).mean().sort_values(by='Survived')
train_df[['Pclass','Survived']].groupby(['Pclass']).mean().sort_values(by='Survived')
train_df[["SibSp","Survived"]].groupby(["SibSp"]).mean().sort_values(by='Survived')
import seaborn as sns

import matplotlib.pyplot as plt

sns.FacetGrid(train_df,col='Survived',row='Sex').map(plt.hist,'Age',bins=20)
sns.catplot(x='Pclass',y='Survived',kind="violin",data=train_df)
sns.catplot(x='Pclass',y='Age',kind='violin',data=train_df)
graf=sns.FacetGrid(train_df,col='Survived',row='Pclass')

graf.map(plt.hist,'Age',bins=20)

graf.add_legend()
sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6).map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
train_df.Age.count()
train_df.loc[train_df['Age']==80]
sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6).map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
print("Before pre-processing train data shape:{}".format(train_df.shape))

print('*'*50)

train_df,test_df=train_df.drop(['Ticket','Cabin'],axis=1),test_df.drop(['Ticket','Cabin'],axis=1)

df=[train_df,test_df]

print('After pre-processing train data shape:{}'.format(train_df.shape))
for i in df:

    i['Title']=i.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'],train_df['Sex'])
for i in df:

    i['Title']=i['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Rev','Sir'],'Uncommon')

    i['Title']=i['Title'].replace(['Mlle','Ms'],'Miss')

    i['Title']=i['Title'].replace('Mme','Mrs')
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
train_df.Title.value_counts().plot(kind='pie')
title_compress={"Mr":1,

               "Miss":2,

               "Mrs":3,

               "Master":4,

               "Uncommon":5}
for i in df:

    i.Title=i.Title.map(title_compress)

    i.Title=i.Title.fillna(0)

train_df.Title
train_df
sex_compress={"female":1,"male":0}
for i in df:

    i.Sex=i.Sex.map(sex_compress).astype(int)
train_df.head()
guess_ages=np.zeros((2,3))
#Sex: 0(Male)/1(Female)

#Pclass: 1,2,3

for dataset in df:

    for i in range(0,2): #sex

        for j in range(0,3): #pclass

            filled_df=dataset[(dataset['Sex'] == i)&(dataset['Pclass'] == j+1)]['Age'].dropna()



            age_guess=filled_df.median()

            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0,2):

        for j in range(0,3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)    
train_df.tail()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
def age_categ(train_df):

    train_df.loc[train_df.Age<=16,'Age']=0

    train_df.loc[(train_df.Age>16) & (train_df.Age<=32),'Age']=1

    train_df.loc[(train_df.Age>32) & (train_df.Age<=48),'Age']=2

    train_df.loc[(train_df.Age>48) & (train_df.Age<=64),'Age']=3

    train_df.loc[(train_df.Age>64) & (train_df.Age<=80),'Age']=4
age_categ(train_df)

age_categ(test_df)
train_df
train_df['Embarked'] = train_df['Embarked'].fillna(train_df.Embarked.dropna().mode()[0])

test_df['Embarked'] = test_df['Embarked'].fillna(test_df.Embarked.dropna().mode()[0])

#train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df.Embarked.isnull().sum()

test_df.Embarked.isnull().sum()
train_df.Embarked.loc[train_df.Embarked=='S']=0

train_df.Embarked.loc[train_df.Embarked=='C']=1

train_df.Embarked.loc[train_df.Embarked=='Q']=2

#embarked_compress={'S': 0, 'C': 1, 'Q': 2}

#train_df['Embarked'].map(embarked_compress).astype(int)
test_df.Embarked.loc[test_df.Embarked=='S']=0

test_df.Embarked.loc[test_df.Embarked=='C']=1

test_df.Embarked.loc[test_df.Embarked=='Q']=2



train_df
train_df
train_df['FareBand']=pd.qcut(train_df['Fare'],4)

train_df.groupby(['FareBand'])['Survived'].mean().to_frame()
train_df
def fare_categ(train_df):

    train_df.loc[train_df.Fare<=7.91,'Fare']=0

    train_df.loc[(train_df.Fare>7.91)&(train_df.Fare<=14.454),'Fare']=1

    train_df.loc[(train_df.Fare>14.454)&(train_df.Fare<= 31.0),'Fare']=2    

    train_df.loc[(train_df.Fare>  31.0)&(train_df.Fare<=512.329),'Fare']=3

    return train_df
fare_categ(train_df)
fare_categ(test_df)
train_df=train_df.drop(['Name','AgeBand','FareBand'],axis=1)

train_df
Y_train=train_df['Survived']

X_train=train_df.drop(['Survived'],axis=1)

X_test  = test_df.drop(["PassengerId","Name"], axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape

X_train
X_test
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

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
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })