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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test_df=test.copy() #used in prediction

train.head()
#Feature info
train.info()
print('-'*40)
test.info()
print(train.isnull().sum())
test.isnull().sum()
print(round(train['Cabin'].isnull().sum()/train.shape[0]*100,2) , 
      "% values of 'Cabin' feature is not filled in the train set ")
print(round(test['Cabin'].isnull().sum()/test.shape[0]*100,2) , 
      "% values of 'Cabin' feature is not filled in the test set ")
#Dropping "Cabin" feature from dataset
train=train.drop('Cabin',axis=1)
test=test.drop('Cabin',axis=1)
print(round(train['Age'].isnull().sum()/train.shape[0]*100,2) , 
      "% values of 'Age' feature is not filled in the train set ")
print(round(test['Age'].isnull().sum()/test.shape[0]*100,2) , 
      "% values of 'Age' feature is not filled in the test set ")
#imputing for embarked in train set and fare in test set
train['Embarked']=train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Fare']=test['Fare'].fillna(train['Fare'].mean())
train.columns #digging deep in to each and every features
train.describe()
#Survived feature analysis
data=pd.DataFrame(train['Survived'].value_counts())
print(data)
sns.barplot(data.index,data['Survived'])
plt.legend()
plt.show()
#Passenger class feature analysis
data=pd.DataFrame(train['Pclass'].value_counts())
print(data)
sns.barplot(data.index,data['Pclass'])
plt.legend()
plt.show()
train['Name'][:5] #Title can be extracted from name which can help us in predicting survival
#Passenger's Sex feature analysis
data=pd.DataFrame(train['Sex'].value_counts())
print(data)
sns.barplot(data.index,data['Sex'])
plt.legend()
plt.show()
#Age feature analysis
print("Passengers with more than 60 years of age is",len(train['Age'][train['Age']>60]))
train['Age'].hist(bins=10)

# Sibsp Feature
print(len(train[train['SibSp']<1])," passengers are without siblings or spouses ")
sns.distplot(train['SibSp'])
# Parch Feature
print(len(train[train['Parch']<1])," passengers are without Parents or child ")
(train['Parch']).hist(bins=20)
print(len(train[(train['SibSp']==0) & (train['Parch']==0)]), "of 891 are travelling single")
# Fare feature
sns.distplot(train['Fare'],bins=20) 
plt.show()                                #We have outliers
train['Fare'].describe()
train['Fare'].quantile(0.82) 
#Embarked feature
data=pd.DataFrame(train['Embarked'].value_counts())
print(data)
sns.barplot(data.index,data['Embarked'])
plt.show()

train.corr()
#Let's compare our features with the target variable Survived feature.
sns.barplot(train['Survived'],train['Sex'])

#Survived vs Embarked
sns.barplot(train['Survived'],train['Embarked'])
#Survived vs Pclass
sns.barplot(train['Survived'],train['Pclass'])
#Survived vs Fare
sns.barplot(train['Survived'],train['Fare'])
#Multivariate analysis
s=sns.FacetGrid(train, col='Survived')
s.map(plt.hist, 'Age', bins=20)
s=sns.FacetGrid(train, col='Survived')
s.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train, col='Survived', row='Pclass')
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
sns.barplot(train['Pclass'],train['Survived'],hue=train['Sex'])

train.pivot_table( index=['Pclass','Sex'], columns='Survived',values='Fare' ,aggfunc='count')


s = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
s.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
s.add_legend()

train.head()
#Dropping useless features 'passengerId','Ticket'
train=train.drop(['PassengerId','Ticket'],axis=1)
test=test.drop(['PassengerId','Ticket'],axis=1)

#we shall apply conversion to both train and test set
com = [train, test] #creating a list

# Extracting title from Name feature
for df in com:
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])
train['Title'].value_counts()
#compressing title feature
for df in com:
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
#title vs survived
sns.barplot(train['Title'],train['Survived'])
# encoding title feature
title_mapp = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for df in com:
    df['Title'] = df['Title'].map(title_mapp)
    df['Title'] = df['Title'].fillna(0)
#remove name feature 

train=train.drop('Name',axis=1)
test=test.drop('Name',axis=1)

#Encoding Sex feature
train['Sex']=train['Sex'].map({'female':1,'male':0}).astype(int)
test['Sex']=test['Sex'].map({'female':1,'male':0}).astype(int)


#We will guess age with sex and Pclass features and impute the values
guess_ages = np.zeros((2,3))
guess_ages
com=[train,test]
for dataset in com:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train.head()
test.info()
test.isnull().sum().sum()
train['AgeBand'] = pd.cut(train['Age'], 4)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
com=[train,test]
for dataset in com:    
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 40), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 60) , 'Age'] = 3
    
train.head()
#dropping age band feature, we already imputed age bands to 'Age' feature
train = train.drop(['AgeBand'], axis=1)
 # Extracting family size
com = [train, test]
for dataset in com:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#We can create another feature called 'is alone'
com=[train, test]
for dataset in com:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# We can drop Parch,Sibsp and family size features in favor of "is alone"
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
com = [train, test]

train.info()

test.info()
com=[train,test]
for dataset in com:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()
#We can now create Fare band
train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
com=[train, test]
for dataset in com:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
train.head(5)
test.head(2)
tr=train.copy()
te=test.copy()
#Label encoding version
X_train=train.drop('Survived',axis=1)
Y_train=train['Survived']
X_test=test
X_train.head(2)
X_test.head(2)

logreg=LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log = logreg.predict(X_test)
acc=cross_val_score(logreg, X_train, Y_train, cv=5,scoring='accuracy')
acc_log = round(acc.mean() * 100, 2)

acc_log

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
acc=cross_val_score(svc, X_train, Y_train, cv=5,scoring='accuracy')
acc_svc = round(acc.mean() * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors =6)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc=cross_val_score(knn, X_train, Y_train, cv=5,scoring='accuracy')
acc_knn = round(acc.mean() * 100, 2)

acc_knn
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gaussian = gaussian.predict(X_test)
acc=cross_val_score(gaussian, X_train, Y_train, cv=5,scoring='accuracy')
acc_gc = round(acc.mean() * 100, 2)

acc_gc
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_DT = decision_tree.predict(X_test)
acc=cross_val_score(decision_tree, X_train, Y_train, cv=5,scoring='accuracy')
acc_dt = round(acc.mean() * 100, 2)

acc_dt
gb=RandomForestClassifier()
param={
    'n_estimators' :[150,300,500],
    'max_depth':[7,20,30],
    
}
gs=GridSearchCV(gb,param,cv=5,n_jobs=-1)
cv_fit=gs.fit(X_train,Y_train)
Y_pred_RF=gs.predict(X_test)
acc_rf=gs.best_score_*100
acc_rf
gb=GradientBoostingClassifier()
param={
    'n_estimators' :[100,150],
    'max_depth':[7,11,15],
    'learning_rate':[0.1]
}
gs=GridSearchCV(gb,param,cv=5,n_jobs=-1)
cv_fit=gs.fit(X_train,Y_train)
Y_pred_gb=cv_fit.predict(X_test)

gs.best_params_
acc_gb=gs.best_score_*100
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
               
              'Decision Tree','Gradient Boosting classifier'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_rf, acc_gc,  
              acc_dt,acc_gb]})
models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_RF
    })
#submission.to_csv('Submission_RFo', index=False)
random_forest = RandomForestClassifier(n_estimators=500,max_depth=30)
random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)
importance = random_forest.feature_importances_
importance
test.columns.tolist()
pd.DataFrame(importance,test.columns.tolist())
