import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import VotingClassifier





from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

train.info()
#handling categorical variables plotting

def pl(feat):

    sur=train[train['Survived']==1][feat].value_counts(dropna=False)

    dead=train[train['Survived']==0][feat].value_counts(dropna=False)

    print(pd.DataFrame([sur,dead],index=['survived','dead']).plot(kind='bar',stacked=True,figsize=(10,5),legend=True))
pl('Pclass')
for x in [train,test]:

    x['Title']=x['Name'].str.extract('([A-Za-z]+)[.]',expand=False)

    x['Title']=x['Title'].map({'Mr':0,'Miss':1,'Mrs':2,'Master':3,\

                        'Dr':3,'Rev':3,'Major':3,'Col':3,'Mlle':3,'Jonkheer':3,'Capt':3\

                        ,'Mme':3,'Don':3,'Dona':3,'Sir':3,'Countess':3,'Ms':3,'Lady':3})
pl('Title')

train.drop(['Name'],axis=1,inplace=True)

test.drop(['Name'],axis=1,inplace=True)

for x in [train,test]:

    x['Sex']=x['Sex'].map({'male':0,'female':1})

    
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)

test['Age'].fillna(test.groupby('Title')['Age'].transform('median'),inplace=True)

#handling continuous value in data visualization

res=sns.FacetGrid(train,hue='Survived',aspect=4,xlim=(0,40))

res.map(sns.kdeplot,'Age')

res.add_legend()
res=sns.FacetGrid(train,hue='Survived',aspect=4,xlim=(40,70))

res.map(sns.kdeplot,'Age')

res.add_legend()
for x in [train,test]:

    x.loc[x['Age']<=16,'Age']=0,

    x.loc[(x['Age']>=16) & (x['Age']<=26),'Age']=1,

    x.loc[(x['Age']>=26) & (x['Age']<=36),'Age']=2,

    x.loc[(x['Age']>=36) & (x['Age']<=62),'Age']=3,

    x.loc[x['Age']>62,'Age']=4,
pl('Age')
c1=train[train['Pclass']==1]['Embarked'].value_counts(dropna=False)

c2=train[train['Pclass']==2]['Embarked'].value_counts(dropna=False)

c3=train[train['Pclass']==3]['Embarked'].value_counts(dropna=False)

pd.DataFrame([c1,c2,c3],index=['c1','c2','c3']).plot(kind='bar',stacked=True,figsize=(10,5),legend=True)
for x in [train,test]:

    x['Embarked']=x['Embarked'].fillna('S')
for x in [train,test]:

    x['Embarked']=x['Embarked'].map({'S':0,'C':1,'Q':2})
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)

test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)
res=sns.FacetGrid(train,hue='Survived',aspect=4,xlim=(0,100))

res.map(sns.kdeplot,'Fare')

res.add_legend()
for x in [train,test]:

    x.loc[x['Fare']<=17,'Fare']=0

    x.loc[(x['Fare']>17) & (x['Fare']<=30),'Fare']=1

    x.loc[(x['Fare']>30) & (x['Fare']<=100),'Fare']=2

    x.loc[x['Fare']>100,'Fare']=3
for x in [train,test]:

    x['Cabin']=x['Cabin'].str[:1]
train['Cabin'].value_counts()
c1=train[train['Pclass']==1]['Cabin'].value_counts()

c2=train[train['Pclass']==2]['Cabin'].value_counts()

c3=train[train['Pclass']==3]['Cabin'].value_counts()

pd.DataFrame([c1,c2,c3],index=['c1','c2','c3']).plot(kind='bar',stacked=True,figsize=(10,5),legend=True)
for x in [train,test]:

    x['Cabin']=x['Cabin'].map({'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2,'G':2.4,'T':2.8})
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)

test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'),inplace=True)

train['Familysize']=train['SibSp']+train['Parch']+1

test['Familysize']=test['SibSp']+test['Parch']+1
train['Familysize'].value_counts()

test['Familysize'].value_counts()
for x in [train,test]:

    x['Familysize']=x['Familysize'].map({1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,6:2.4,7:2.6,8:3,11:3.4})
X_train=train.drop(['PassengerId','SibSp','Parch','Ticket','Survived'],axis=1)

X_test=test

X_test=X_test.drop(['PassengerId','SibSp','Parch','Ticket'],axis=1)

y_train=train['Survived']
ran = RandomForestClassifier(random_state=1)

knn = KNeighborsClassifier()

log = LogisticRegression()

gbc = GradientBoostingClassifier()

svc = SVC(probability=True)

ext = ExtraTreesClassifier()

ada = AdaBoostClassifier()

gnb = GaussianNB()

gpc = GaussianProcessClassifier()

bag = BaggingClassifier()





models = [ran, knn, log,  gbc, svc, ext, ada, gnb, gpc, bag]         

scores = []



for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores.append(acc.mean())

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression','Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Score': scores})



result_df = results.sort_values(by='Score', ascending=False).reset_index(drop=True)

result_df
import matplotlib.pyplot as plt

# Plot results

sns.barplot(x='Score', y = 'Model', data = result_df, color = 'c')

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')


Cs = [0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 50, 100]

gammas = [0.001, 0.01, 0.1, 1]



hyperparams = {'C': Cs, 'gamma' : gammas}





gd=GridSearchCV(estimator = SVC(probability=True), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")





gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

n_estimators = [100, 250, 500, 750, 1000, 1250, 1500]



hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}



gd=GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
import numpy as np

penalty = ['l1', 'l2']

C = np.logspace(0, 4, 10)



hyperparams = {'penalty': penalty, 'C': C}



gd=GridSearchCV(estimator = LogisticRegression(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_restarts_optimizer = [0, 1, 2, 3]

max_iter_predict = [1, 2, 5, 10, 20, 35, 50, 100]

warm_start = [True, False]



hyperparams = {'n_restarts_optimizer': n_restarts_optimizer, 'max_iter_predict': max_iter_predict, 'warm_start': warm_start}



gd=GridSearchCV(estimator = GaussianProcessClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators = [10, 25, 50, 75, 100, 125, 150, 200]

learning_rate = [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2]



hyperparams = {'n_estimators': n_estimators, 'learning_rate': learning_rate}



gd=GridSearchCV(estimator = AdaBoostClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

algorithm = ['auto']

weights = ['uniform', 'distance']

leaf_size = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]



hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 

               'n_neighbors': n_neighbors}



gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators = [10, 25, 50, 75, 100]

max_depth = [3, None]

max_features = [1, 3, 5, 7]

min_samples_split = [2, 4, 6, 8, 10]

min_samples_leaf = [2, 4, 6, 8, 10]



hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,

               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}



gd=GridSearchCV(estimator = RandomForestClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators = [10, 25, 50, 75, 100]

max_depth = [3, None]

max_features = [1, 3, 5, 7]

min_samples_split = [2, 4, 6, 8, 10]

min_samples_leaf = [2, 4, 6, 8, 10]



hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,

               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}



gd=GridSearchCV(estimator = ExtraTreesClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators = [10, 15, 20, 25, 50, 75, 100, 150]

max_samples = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 50]

max_features = [1, 3, 5, 7]



hyperparams = {'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features}



gd=GridSearchCV(estimator = BaggingClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
ran = RandomForestClassifier(n_estimators=10,

                             max_depth=3, 

                             max_features=3,

                             min_samples_leaf=2, 

                             min_samples_split=8,  

                             random_state=1)



knn = KNeighborsClassifier(algorithm='auto', 

                           leaf_size=25, 

                           n_neighbors=16, 

                           weights='uniform')  



log = LogisticRegression(C=2.7825594022071245,

                         penalty='l2')





gbc = GradientBoostingClassifier(learning_rate=0.01,

                                 n_estimators=1000,

                                 random_state=1)





svc = SVC(probability=True)



ext = ExtraTreesClassifier(max_depth=None, 

                           max_features=1,

                           min_samples_leaf=4, 

                           min_samples_split=8,

                           n_estimators=10,

                           random_state=1)



ada = AdaBoostClassifier(learning_rate=0.1, 

                         n_estimators=150,

                         random_state=1)



gpc = GaussianProcessClassifier()



bag = BaggingClassifier(random_state=1)





models = [ran, knn, log, gbc, svc, ext, ada, gnb, gpc, bag]         

scores_v3 = []





for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores_v3.append(acc.mean())
results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression','Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Original Score': scores,'Score with tuned parameters': scores_v3})



result_df = results.sort_values(by='Score with tuned parameters', ascending=False).reset_index(drop=True)

result_df.head()
from sklearn.model_selection import cross_validate

grid_hard = VotingClassifier(estimators = [('Random Forest', ran), 

                                           ('Logistic Regression', log),

                                           

                                           ('Gradient Boosting', gbc),

                                           ('Extra Trees', ext),

                                           ('AdaBoost', ada),

                                           ('Gaussian Process', gpc),

                                           ('SVC', svc),

                                           ('K Nearest Neighbour', knn),

                                           ('Bagging Classifier', bag)], voting = 'hard')



grid_hard_cv = cross_validate(grid_hard, X_train, y_train, cv = 10)

grid_hard.fit(X_train, y_train)



print("Hard voting on test set score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
grid_soft = VotingClassifier(estimators = [('Random Forest', ran), 

                                           ('Logistic Regression', log),

                                           

                                           ('Gradient Boosting', gbc),

                                           ('Extra Trees', ext),

                                           ('AdaBoost', ada),

                                           ('Gaussian Process', gpc),

                                           ('SVC', svc),

                                           ('K Nearest Neighbour', knn),

                                           ('Bagging Classifier', bag)], voting = 'soft')



grid_soft_cv = cross_validate(grid_soft, X_train, y_train, cv = 10)

grid_soft.fit(X_train, y_train)



print("Soft voting on test set score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
# Final predictions

predictions = grid_soft.predict(X_test)



output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")