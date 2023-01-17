# This Python 3 environment comes with many helpful analytics libraries installed



# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

sns.set() # setting seaborn default for plots







%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
test_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
train_df['Survived'].value_counts()
def bar_chart(feature):

    survived = train_df[train_df['Survived']==1][feature].value_counts()

    dead = train_df[train_df['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
bar_chart('Pclass')
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
train_test_data = [train_df, test_df] # combining train and test dataset



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train_df['Title'].value_counts()
test_df['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
bar_chart('Title')
X_train_df = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])

X_test_df = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
y_train_df = train_df['Survived']

y_test_df = test_df['PassengerId']
X_train_df.isnull().sum()
X_test_df.isnull().sum()
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
X_train_df['Age'] = X_train_df[['Age','Pclass']].apply(impute_age,axis=1)

X_test_df['Age'] = X_test_df[['Age','Pclass']].apply(impute_age,axis=1)
def Age_cat(x):

    if x <=4 :

        return 1

    elif x>4 and x<=14:

        return 2

    elif x>14 and x<=30:

        return 3

    else:

        return 4
X_train_df['Age'] = X_train_df['Age'].apply(Age_cat)

X_test_df['Age'] = X_test_df['Age'].apply(Age_cat)
X_train_df.Age.unique()
X_train_df['With_someone'] = X_train_df['SibSp'] | X_train_df['Parch']

X_test_df['With_someone'] = X_test_df['SibSp'] | X_test_df['Parch']

X_train_df['Family'] = X_train_df['SibSp'] + X_train_df['Parch']+1

X_test_df['Family'] = X_test_df['SibSp'] + X_test_df['Parch']+1
X_train_df['With_someone'] =X_train_df['With_someone'].apply(lambda x:1 if x >=1 else 0)

X_test_df['With_someone'] =X_test_df['With_someone'].apply(lambda x:1 if x >=1 else 0)
X_train_df['With_someone'].unique()
X_train_df.head()
mod = X_train_df.Embarked.value_counts().argmax()

X_train_df.Embarked.fillna(mod, inplace=True)
fare_med = train_df.Fare.median()

X_test_df.Fare.fillna(fare_med, inplace=True)
X_train_df.isnull().sum()
X_test_df.isnull().sum()
X_train_df.columns
X_train_df.replace({"male": 0, "female": 1}, inplace=True)

X_test_df.replace({"male": 0, "female": 1}, inplace=True)

X_train_df.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)

X_test_df.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)
X_train_df.head()
X_train_df = pd.get_dummies(X_train_df, columns=['Pclass', 'Embarked','Age','Title'], drop_first=True)

X_test_df = pd.get_dummies(X_test_df, columns=['Pclass', 'Embarked','Age','Title'], drop_first=True)

X_train_df.head()
X_train_df = X_train_df.drop(columns=['SibSp','Parch'])

X_test_df = X_test_df.drop(columns=['SibSp','Parch'])
X_train_df.shape, X_test_df.shape
from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler()

X_train_df[['Fare','Family']] = sc_X.fit_transform(X_train_df[['Fare','Family']])

X_test_df[['Fare','Family']] = sc_X.transform(X_test_df[['Fare','Family']])
X_train_df.head()
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier
logi_clf = LogisticRegression(random_state=0)

logi_parm = {"penalty": ['l1', 'l2'], "C": [0.1, 0.5, 1, 5, 10, 50]}



svm_clf = SVC(random_state=0)

svm_parm = {'kernel': ['rbf', 'poly'], 'C': [0.1, 0.5, 1, 5, 10, 50], 'degree': [3, 5, 7], 

            'gamma': ['auto', 'scale']}



dt_clf = DecisionTreeClassifier(random_state=0)

dt_parm = {'criterion':['gini', 'entropy']}



knn_clf = KNeighborsClassifier()

knn_parm = {'n_neighbors':[5, 10, 15, 20], 'weights':['uniform', 'distance'], 'p': [1,2]}



gnb_clf = GaussianNB()

gnb_parm = {'priors':['None']}



clfs = [logi_clf, svm_clf, dt_clf, knn_clf]

params = [logi_parm, svm_parm, dt_parm, knn_parm] 
clf1 = RandomForestClassifier()

clf1.fit(X_train_df,y_train_df)

rf_rand = GridSearchCV(clf1,{'n_estimators':[50,100,200,300,500],'max_depth':[i for i in range (2,11)]},cv=10)

rf_rand.fit(X_train_df,y_train_df)

print(rf_rand.best_score_)

print(rf_rand.best_params_)
clf2 = GradientBoostingClassifier()

clf2.fit(X_train_df,y_train_df)

gb_rand = GridSearchCV(clf2,{'n_estimators':[50,100,200,300,500],'learning_rate':[0.01,0.1,1],'max_depth':[i for i in range (2,11)]},cv=10)

gb_rand.fit(X_train_df,y_train_df)

print(gb_rand.best_score_)

print(gb_rand.best_params_)
clf3 = SVC(gamma='auto')

clf3.fit(X_train_df,y_train_df)

svc_rand = GridSearchCV(clf3,{'C':[5,10,15,20],'degree':[i for i in range(1,11)]},cv=10)

svc_rand.fit(X_train_df,y_train_df)

print(svc_rand.best_score_)

print(svc_rand.best_params_)
clf1 = RandomForestClassifier(max_depth=6,n_estimators=200)

clf1.fit(X_train_df,y_train_df)

clf2 = GradientBoostingClassifier(n_estimators=300,learning_rate=0.01,max_depth=4,random_state=0)

clf2.fit(X_train_df,y_train_df)

clf3 = SVC(C=5,degree=1,gamma='auto',probability=True)

clf3.fit(X_train_df,y_train_df)
eclf = VotingClassifier(estimators=[('rf',clf1),('gb',clf2),('svc',clf3)],voting='soft',weights=[2.5,2.5,2])
eclf.fit(X_train_df,y_train_df)
#clfs_opt = []

#clfs_best_scores = []

#clfs_best_param = []

#for clf_, param in zip(clfs, params):

#    clf = RandomizedSearchCV(clf_, param, cv=5)

#    clf.fit(X_train_sc, y_train_df)

#    clfs_opt.append(clf)

#    clfs_best_scores.append(clf.best_score_)

#    clfs_best_param.append(clf.best_params_)
#max(clfs_best_scores)
#arg = np.argmax(clfs_best_scores)

#clfs_best_param[arg]
#clf = clfs_opt[arg]
#pred = clf.predict(X_test_sc)
#Grad_clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0)

#Grad_clf.fit(X_train_df,y_train_df)
#rand = RandomizedSearchCV(Grad_clf,{'learning_rate':[0.01,0.1,1],'max_depth':[1,5,10],

                                    #'n_estimators':[50,100,200,500]},n_iter=15,cv=10)
#rand.fit(X_train_df,y_train_df)
#print(rand.best_score_)

#print(rand.best_params_)

#Grad_clf = GradientBoostingClassifier(n_estimators=200,learning_rate=0.01,max_depth=5)

#Grad_clf.fit(X_train_df,y_train_df)
pred = eclf.predict(X_test_df)
cols = ['PassengerId', 'Survived']

submit_df = pd.DataFrame(np.hstack((y_test_df.values.reshape(-1,1),pred.reshape(-1,1))), 

                         columns=cols)
submit_df.to_csv('submission.csv', index=False)
submit_df.head()