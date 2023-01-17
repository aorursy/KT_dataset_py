import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

import seaborn as sns

%matplotlib inline

data_train = pd.read_csv('../input/titanic/train.csv')

data_train.head()

data_train.describe()
data_train = data_train.drop(['PassengerId','Name','Ticket'],axis = 'columns')



print(data_train.isnull().sum())

plt.figure(figsize=(20,10))

sns.heatmap(data_train.isnull())
data_train = data_train.drop('Cabin',axis='columns')
plt.figure(figsize=(20,15))

for i, col in enumerate(data_train.columns):

    plt.subplot(2,4,i+1)

    sns.boxplot(x=col,y='Age', data=data_train)
plt.subplot(1,2,1)

sns.boxplot('Pclass','SibSp',data=data_train)

plt.subplot(1,2,2)

sns.boxplot('Pclass','Parch',data=data_train)
print(data_train[['Age','Pclass']].groupby('Pclass').mean())
# 25 29 37

def AgeMissing(age_null):

    age = age_null['Age'].tolist()

    pclass = age_null['Pclass'].tolist()

    

    for i in range(len(age)):

        if pd.isnull(age[i]):

            if pclass[i] == 3:

                age[i] = 25

            elif pclass[i] == 2:

                age[i] = 29

            else:

                age[i] = 38

    return age



data_train.Age = AgeMissing(data_train[['Age','Pclass']])
data_train["Embarked"].value_counts()
data_train["Embarked"] = data_train["Embarked"].fillna('S')

data_train.isnull().sum()
grid = sns.FacetGrid(data_train, col='Survived')

grid.map(plt.hist,'Age',bins=15)
grid = sns.FacetGrid(data_train, col='Survived',row='Pclass')

grid.map(plt.hist,'Age',bins=15)
data_train.head()
sns.catplot('Sex',kind='count',data=data_train,hue='Survived')

sns.catplot('Sex',kind='count',data=data_train,hue='Pclass',col='Survived')
data_train.head()
data_train = data_train.join(pd.get_dummies(data_train.Embarked,drop_first =True).rename(columns = {'Q':'Embarked_Q',

                                                      'S':'Embarked_S'}))

data_train = data_train.drop('Embarked',axis='columns')



data_train.Sex = data_train.Sex.apply(lambda x: 1 if x == 'male' else 0)

data_train.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



x = data_train.columns.tolist()

y='Survived'

x.remove(y)



x_train,x_test,y_train,y_test =  train_test_split(data_train[x],data_train[y],test_size=0.2)
lg = LogisticRegression()

lg.fit(x_train,y_train)

lg_predicts = lg.predict(x_test)



from sklearn.metrics import classification_report,confusion_matrix

print('Confusion matrix for Logistic :\n ',confusion_matrix(y_test,lg_predicts),"\n")

print(classification_report(y_test,lg_predicts))
from sklearn.tree import DecisionTreeClassifier

tr = DecisionTreeClassifier()

tr.fit(x_train,y_train)

tr_predicts = tr.predict(x_test)



print('Confusion matrix for Decision Tree:\n ',confusion_matrix(y_test,tr_predicts),"\n")

print(classification_report(y_test,tr_predicts))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train,y_train)

rf_predicts = rf.predict(x_test)



print('Confusion matrix for Random Forest:\n ',confusion_matrix(y_test,rf_predicts),"\n")

print(classification_report(y_test,rf_predicts))
plt.figure(figsize=(18,8))

sns.heatmap(data_train.corr(),center=0,annot=True)
data_train.Age.describe()
data_train.Age = data_train.Age.apply(lambda x : 'Age_10' if x<11 else ('Age_20' if x<21 else ('Age_40' if x<41 else 'Age_above_40')))

#data_train.Age = data_train.Age.apply(lambda x : 1 if x<11 else (2 if x<21 else (3 if x<41 else 0)))



data_train = data_train.join(pd.get_dummies(data_train.Age,drop_first=True))

data_train = data_train.drop('Age',axis='columns')

x = data_train.columns.tolist()

y='Survived'

x.remove(y)



x_train,x_test,y_train,y_test =  train_test_split(data_train[x],data_train[y],test_size=0.2)
lg = LogisticRegression()

lg.fit(x_train,y_train)

lg_predicts = lg.predict(x_test)



print('Confusion matrix for Logistic :\n ',confusion_matrix(y_test,lg_predicts),"\n")

print(classification_report(y_test,lg_predicts))
rf = RandomForestClassifier()

rf.fit(x_train,y_train)

rf_predicts = rf.predict(x_test)



print('Confusion matrix for Random Forest:\n ',confusion_matrix(y_test,rf_predicts),"\n")

print(classification_report(y_test,rf_predicts))
from sklearn.model_selection import KFold,cross_val_score

rf = RandomForestClassifier(n_estimators = 100)



x = data_train.columns.tolist()

y='Survived'

x.remove(y)



print("Recall - " , np.mean(cross_val_score(rf,data_train[x],data_train[y],cv=10,scoring='recall')))

print("precision - " , np.mean(cross_val_score(rf,data_train[x],data_train[y],cv=10,scoring='precision')))

print("accuracy - " , np.mean(cross_val_score(rf,data_train[x],data_train[y],cv=10,scoring='accuracy')))



x = data_train[x]

y = data_train[y]



#x = pd.DataFrame(scaler.fit_transform(x),columns= x.columns)





scores=[]



cv = KFold(n_splits=10, random_state=42, shuffle=False)

for train_index, test_index in cv.split(x):

    x_train, x_test, y_train, y_test = x.loc[train_index], x.loc[test_index], y.loc[train_index], y.loc[test_index]

    rf.fit(x_train, y_train)

    scores.append(rf.score(x_test, y_test))

    



np.mean(scores)

data_train = pd.read_csv('../input/titanic/train.csv')

data_test = pd.read_csv('../input/titanic/test.csv')

passenger_id = data_test.PassengerId

def AgeMissing(age_null):

    age = age_null['Age'].tolist()

    pclass = age_null['Pclass'].tolist()

    

    for i in range(len(age)):

        if pd.isnull(age[i]):

            if pclass[i] == 3:

                age[i] = 25

            elif pclass[i] == 2:

                age[i] = 29

            else:

                age[i] = 38

    return age



def GetTitle(name):

    title = re.findall(' ([A-Za-z]+)\.', name)[0]

    if title:

        return title

    else:

        return("")

def ChangeFeatures(data):

    data.Age = AgeMissing(data[['Age','Pclass']])

    data["Embarked"] = data["Embarked"].fillna('S')

    data['Fare'].fillna(np.mean(data['Fare']))

    data["Embarked"] = data["Embarked"].map({'S':0,'C':1,'Q':2})

    data.Sex = data.Sex.apply(lambda x: 1 if x=='male' else 0)

    

    # Get the title from the name and arrange them in a numeric categorial format

    data['Title'] = data.Name.apply(GetTitle)

    data['Title'] = data['Title'].replace('Ms','Miss')

    data['Title'] = data['Title'].replace(['Mlle','Mme'],'Mrs')

    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 

                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5})

    data['Title'].fillna(0)

    

    # adding Parch and SibSp  as a family size and SibSp will be dropped.

    data['FamilySize'] = data.Parch + data.SibSp

    

    # Changing the Fare values to a numeric categorial of a respective ranges

    bins = [0,7.91,14.45,31,513]

    labels = [0,1,2,3]

    data.Fare = pd.cut(data.Fare,bins,labels=labels).astype(int)

    

    # Changing the Age values to a numeric categorial of a respective ranges

    bins = [0,6,16,29,42,55,80]

    labels = [0,1,2,3,4,5]

    data.Age = pd.cut(data.Age,bins,labels=labels).astype(int)



    data.drop(['Name','Cabin','PassengerId','Ticket','SibSp'],axis='columns',inplace=True)

    return(data)



data_train = ChangeFeatures(data_train)

data_test = ChangeFeatures(data_test)
plt.figure(figsize=(20,10))

sns.heatmap(data_train.corr(),annot=True)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm  import SVC

from sklearn.ensemble import GradientBoostingClassifier



x = data_train.columns.tolist()

y = 'Survived'

x.remove(y)

knn = KNeighborsClassifier(n_neighbors = 7)



rf = RandomForestClassifier(n_jobs= -1,

                            n_estimators= 100,

                            warm_start = True,

                            max_depth =  6,

                            min_samples_leaf = 2,

                            max_features = 'auto')



svc = SVC(gamma=.1, kernel='rbf', probability=True)



gb = GradientBoostingClassifier(n_estimators = 100,

                                max_depth = 5,

                                min_samples_leaf = 2

                               )





print("\naccuracy RF - " , np.mean(cross_val_score(rf,data_train[x],data_train[y],cv=10,scoring='accuracy')))

print("\naccuracy KNN - " , np.mean(cross_val_score(knn,data_train[x],data_train[y],cv=10,scoring='accuracy')))

print("\naccuracy SVC - " , np.mean(cross_val_score(svc,data_train[x],data_train[y],cv=10,scoring='accuracy')))

print("\naccuracy GB - " , np.mean(cross_val_score(gb,data_train[x],data_train[y],cv=10,scoring='accuracy')))
# kfold with 5 folds is used for training 

n_folds = 5

cv = KFold(n_splits=n_folds, random_state=42, shuffle=False)

def PerformCV(clf,test):

    

    local_test=np.empty((n_folds,len(data_test)))

    train_labels = np.zeros(len(data_train)) +2 

    i=0

    for train_index, test_index in cv.split(data_train[x]):

        x_train = data_train[x].loc[train_index]

        x_test = data_train[x].loc[test_index]

        y_train = data_train[y].loc[train_index]

        

        clf.fit(x_train, y_train)

        train_labels[test_index] = clf.predict(x_test)

        local_test[i,:] =  clf.predict(test)

        i += 1

    return (train_labels,np.mean(local_test,axis=0))





rf_train,rf_test = PerformCV(rf,data_test)

knn_train,knn_test = PerformCV(knn,data_test)

svc_train,svc_test = PerformCV(svc,data_test)

gb_train,gb_test = PerformCV(gb,data_test)
# concatenation

x_train = pd.DataFrame({'KNN' : knn_train,

              'RF' : rf_train,

              'SVC':svc_train,

              'GB':gb_train})



x_test = pd.DataFrame({'KNN' : knn_test,

              'RF' : rf_test,

              'SVC':svc_test,

              'GB':gb_test})



y_train = data_train[y]
import xgboost as xgb

gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1)

gbm.fit(x_train, y_train)

predictions = gbm.predict(x_test)

predictions
final_output = pd.DataFrame({'PassengerId' : passenger_id,

              'Survived':predictions})

final_output.to_csv("titanic_predictions.csv",index=False)

final_output.head()