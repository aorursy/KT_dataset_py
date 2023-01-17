import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data_train= pd.read_csv("../input/train.csv")
data_test=pd.read_csv("../input/test.csv")
data_train.head()
#checking missing data
sns.heatmap(data_train.isnull(), yticklabels= False)
sns.countplot(x='Survived',data=data_train)
sns.countplot(x='Survived', hue='Sex',data= data_train)
sns.countplot(x='Survived', hue='Pclass', data= data_train)
sns.set_style("whitegrid")
sns.boxplot(x='Pclass', y='Age', data= data_train)
#avg of class 1 age= 38
#avg of class 2 age= 29
#avg of class 3 age= 24

def fill_age(a):
    Age= a[0]
    Pclass= a[1]
    if(pd.isnull(Age)):
        if(Pclass ==1):
            return 38
        if(Pclass==2):
            return 29
        if(Pclass==3):
            return 24
    else:
        return Age
data_train['Age']= data_train[['Age','Pclass']].apply(fill_age,axis=1)
#checking missing data again
sns.heatmap(data_train.isnull(), yticklabels= False)
combine = [data_train, data_test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(data_train['Title'], data_train['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

data_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

data_train.head()
data_train['family']= data_train['SibSp']+data_train['Parch']
data_train.head()
#Cabin has a lot of missing values, so, let's drop that column
#Ticket is also not of any use to us
data_train.drop(['Cabin','Name','Ticket','PassengerId','SibSp','Parch'], inplace= True, axis=1)
data_train.head()
data_train.info()
#converting objects to numerical values
sex = pd.get_dummies(data_train['Sex'],drop_first=True)
embark = pd.get_dummies(data_train['Embarked'],drop_first=True)
data_train.drop(['Sex','Embarked'],axis=1,inplace=True)
data_train=pd.concat([data_train,sex,embark],axis=1)
data_train.head()
data_train.info()
sns.heatmap(data_train.corr())
data_test.head()
data_test['family']= data_test['SibSp']+data_test['Parch']
data_test.drop(['Cabin','Name','Ticket','PassengerId','SibSp','Parch'], inplace= True, axis=1)
sex = pd.get_dummies(data_test['Sex'],drop_first=True)
embark = pd.get_dummies(data_test['Embarked'],drop_first=True)
data_test.drop(['Sex','Embarked'],axis=1,inplace=True)
data_test=pd.concat([data_test,sex,embark],axis=1)
data_test.head()
data_test.info()
data_test['Fare'].fillna(value=data_test['Fare'].mean(),inplace=True)
data_test.info()
sns.set_style("whitegrid")
sns.boxplot(x='Pclass', y='Age', data= data_test)
#avg of class 1 age= 42
#avg of class 2 age= 27
#avg of class 3 age= 24

def fill_age(a):
    Age= a[0]
    Pclass= a[1]
    if(pd.isnull(Age)):
        if(Pclass ==1):
            return 42
        if(Pclass==2):
            return 26
        if(Pclass==3):
            return 24
    else:
        return Age
data_test['Age']= data_test[['Age','Pclass']].apply(fill_age,axis=1)
data_test.info()
from sklearn.model_selection import train_test_split
X= data_train.drop("Survived",axis=1)
Y= data_train["Survived"] 
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, 
                                                    random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics 
models=[]
models.append(('DTC',DecisionTreeClassifier()))
models.append(('SVM',SVC()))
models.append(('GNB',GaussianNB()))
models.append(('KNC',KNeighborsClassifier()))
models.append(('LR',LogisticRegression()))
models.append(('RFC',RandomForestClassifier()))
models.append(("MLP",MLPClassifier()))
models.append(("GBC",GradientBoostingClassifier()))
#checking accuracy score of different classification algorithms
cams=[]
for cam,model in models:
    model.fit(X_train,y_train)
    pre= model.predict(X_test)
    g= metrics.accuracy_score(pre,y_test)
    print("%s: %f "%(cam, g))
#Random Forest Classifier gives best accuracy score
rm= GradientBoostingClassifier()

rm.fit(X_train,y_train)
prediction = rm.predict(data_test)
print(prediction)
test= pd.read_csv("../input/test.csv")
submission= pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":prediction})
submission.to_csv('submission.csv',index=False)
submission= pd.read_csv('submission.csv')
submission.head()
