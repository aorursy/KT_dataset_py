import numpy as np

import pandas as pd

import os

#print(os.listdir('../input/'))

train_data=pd.read_csv('../input/train.csv')

train_data.info()
train_data.isnull().sum()
test_data=pd.read_csv('../input/test.csv')

test_data.isnull().sum()
import seaborn as sns

a=sns.barplot(x='Pclass',y='Survived',data=train_data)
b=a=sns.barplot(x='Sex',y='Survived',data=train_data)
train_data['Family Size']=train_data['SibSp']+train_data['Parch']+1

test_data['Family Size']=test_data['SibSp']+test_data['Parch']+1



fs={1:"Alone",2:"Small",3:"Large",4:"Large",5:"Large",6:"Large",7:"Large",8:"Large",9:"Large",10:"Large",11:"Large"}



train_data['Family Size']=train_data['Family Size'].map(fs)

test_data['Family Size']=test_data['Family Size'].map(fs)



#m={"Alone":1,"Small":2,"Large":3}



#train_data['Family Size']=train_data['Family Size'].map(m)

#test_data['Family Size']=test_data['Family Size'].map(m)







#train_data['is_alone'] = [1 if i<2 else 0 for i in train_data['Family Size']]

#test_data['is_alone'] = [1 if i<2 else 0 for i in test_data['Family Size']]
print('Oldest Passenger Age:',train_data['Age'].max(),'Years')

print('Youngest Passenger Age:',train_data['Age'].min(),'Years')

print('Average Passenger Age:',train_data['Age'].mean(),'Years')

total=[train_data,test_data]

for i in total:

    i['Title'] = i['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 0, "Rev": 0, "Col": 0, "Major": 0, "Mlle": 1,"Countess": 2,

                 "Ms": 1, "Lady": 2, "Jonkheer": 3, "Don": 1, "Dona" : 2, "Mme": 1,"Capt": 0,"Sir": 1 }

print(train_data['Title'].value_counts())

print('------------')

print(test_data['Title'].value_counts())

for i in total:

    i['Title'] = i['Title'].map(title_mapping)

print(train_data.groupby('Title')['Age'].mean())

print('-------')

print(test_data.groupby('Title')['Age'].mean())
#train_data[train_data.Age.isnull()]

train_data.loc[(train_data.Age.isnull())&(train_data.Title==0),'Age']=33

train_data.loc[(train_data.Age.isnull())&(train_data.Title==1),'Age']=22

train_data.loc[(train_data.Age.isnull())&(train_data.Title==2),'Age']=36

train_data.loc[(train_data.Age.isnull())&(train_data.Title==3),'Age']=5



test_data.loc[(test_data.Age.isnull())&(test_data.Title==0),'Age']=32

test_data.loc[(test_data.Age.isnull())&(test_data.Title==1),'Age']=22

test_data.loc[(test_data.Age.isnull())&(test_data.Title==2),'Age']=40

test_data.loc[(test_data.Age.isnull())&(test_data.Title==3),'Age']=7



#train_data.head(20)





def age_group_fun(age):

    a= ''

    if age <= 10:

        a = "child"

    elif age <= 22:

        a = "teenager"

    elif age <= 33:

        a = "yong_Adult"

    elif age <= 45:

        a = "middle_age"

    else:

        a = "old"

    return a



train_data['age_group'] =train_data['Age'].map(age_group_fun)

test_data['age_group'] = test_data['Age'].map(age_group_fun)
train_data.head(5)
train_data=train_data.drop(['Ticket'],axis=1)

test_data=test_data.drop(['Ticket'],axis=1)
train_data=train_data.drop(['Cabin'],axis=1)

test_data=test_data.drop(['Cabin'],axis=1)
train_data=train_data.drop(['Name'],axis=1)

test_data=test_data.drop(['Name'],axis=1)

train_data.head(15)
test_data[test_data.Fare.isnull()]
mv=test_data[(test_data.Pclass==3)&(test_data.Embarked=='S')& (test_data.Sex=='male')].Fare.mean()

print(mv)

test_data.Fare.fillna(mv,inplace=True)

#train_data['Age']=train_data['Age'].astype(int)

#train_data['Fare']=train_data['Fare'].astype(int)

#test_data['Fare']=test_data['Fare'].astype(int)

#test_data['Age']=test_data['Age'].astype(int)
def fare_group_fun(fare):

    a= ''

    if fare <= 7.896:

        a = "low"

    elif fare <= 14.454:

        a = "normal"

    elif fare <= 30.696:

        a = "middle"

    else:

        a = "high"

    return a



train_data['fare_group'] = train_data['Fare'].map(fare_group_fun)

test_data['fare_group'] = test_data['Fare'].map(fare_group_fun)
train_data['Pclass'].astype(str)

train_data['Sex'].astype(str)

#train_data['is_alone'].astype(str)

test_data['Pclass'].astype(str)

test_data['Sex'].astype(str)

#test_data['is_alone'].astype(str)
target=train_data['Survived']

train_data=train_data.drop(['Survived'],axis=1)

print(train_data.shape,target.shape)
train_data[train_data.Embarked.isnull()]
#sex={"male":0,"female":1}

#train_data['Sex']=train_data['Sex'].map(sex)

#test_data['Sex']=test_data['Sex'].map(sex)
test_data.head(5)
#emb={"S":0,"C":1,"Q":2}

#train_data['Embarked']=train_data['Embarked'].fillna('S')

#test_data['Embarked']=test_data['Embarked'].fillna('S')



#train_data['Embarked']=train_data['Embarked'].map(emb)

#test_data['Embarked']=test_data['Embarked'].map(emb)
train_data.head(5)

test_data.head(5)
# onehot encoding & drop unused variables

sex= pd.get_dummies(train_data['Sex'],drop_first=True)

Embarked=pd.get_dummies(train_data['Embarked'],drop_first=True)

Pclass=pd.get_dummies(train_data['Pclass'],drop_first=True)

Title=pd.get_dummies(train_data['Title'],drop_first=True)

#is_alone=pd.get_dummies(train_data['is_alone'],drop_first=True)

age_group=pd.get_dummies(train_data['age_group'],drop_first=True)

fare_group=pd.get_dummies(train_data['fare_group'],drop_first=True)

#fare_group=pd.get_dummies(train_data['fare_group'],drop_first=True)

family_size=pd.get_dummies(train_data['Family Size'],drop_first=True)

train_data=pd.concat([train_data,sex,Embarked,Pclass,Title,age_group,fare_group,family_size],axis=1)



train_data=train_data.drop(['Sex'],axis=1)

train_data=train_data.drop(['Embarked'],axis=1)

train_data=train_data.drop(['Pclass'],axis=1)

train_data=train_data.drop(['Title'],axis=1)

#train_data=train_data.drop(['is_alone'],axis=1)

train_data=train_data.drop(['age_group'],axis=1)

train_data=train_data.drop(['fare_group'],axis=1)

train_data=train_data.drop(['Family Size'],axis=1)

train_data.info()
sex= pd.get_dummies(test_data['Sex'],drop_first=True)

Embarked=pd.get_dummies(test_data['Embarked'],drop_first=True)

Pclass=pd.get_dummies(test_data['Pclass'],drop_first=True)

Title=pd.get_dummies(test_data['Title'],drop_first=True)

#is_alone=pd.get_dummies(train_data['is_alone'],drop_first=True)

age_group=pd.get_dummies(test_data['age_group'],drop_first=True)

#fare_group=pd.get_dummies(test_data['fare_group'],drop_first=True)

fare_group=pd.get_dummies(test_data['fare_group'],drop_first=True)

family_size=pd.get_dummies(test_data['Family Size'],drop_first=True)

test_data=pd.concat([test_data,sex,Embarked,Pclass,Title,age_group,fare_group,family_size],axis=1)



test_data=test_data.drop(['Sex'],axis=1)

test_data=test_data.drop(['Embarked'],axis=1)

test_data=test_data.drop(['Pclass'],axis=1)

test_data=test_data.drop(['Title'],axis=1)

#train_data=train_data.drop(['is_alone'],axis=1)

test_data=test_data.drop(['age_group'],axis=1)

test_data=test_data.drop(['fare_group'],axis=1)

test_data=test_data.drop(['Family Size'],axis=1)

test_data.isnull().sum()
train_data=train_data.drop(['PassengerId'], axis=1)

#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':target})

q=test_data['PassengerId']

test_data=test_data.drop(['PassengerId'], axis=1)



test_data.head(5)

train_data=train_data.drop(['Age'], axis=1)

train_data=train_data.drop(['Fare'], axis=1)

test_data=test_data.drop(['Fare'], axis=1)

test_data=test_data.drop(['Age'], axis=1)
train_data.head(5)
test_data.head(5)
#train_data['Age'].astype(int)

#train_data['Fare'].astype(int)

#from sklearn.model_selection import train_test_split

#X_train,X_test,Y_train,Y_test=train_test_split(train_data,target,test_size=0.2,random_state=0)

#from sklearn.preprocessing import StandardScaler

#sc=StandardScaler()

#X_train=sc.fit_transform(X_train)

#X_test=sc.fit_transform(X_test)


#from sklearn.model_selection import KFold#

##from sklearn.model_selection import cross_val_score

#k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#from sklearn.ensemble import RandomForestClassifier

#clf= RandomForestClassifier(criterion='gini', 

                             #n_estimators=10000,

                             #min_samples_split=10,

                             #min_samples_leaf=1,

                             #max_features='auto',

                             #oob_score=True,

                             #random_state=1,

                             #n_jobs=-1)

#from sklearn.tree import DecisionTreeClassifier

#clf=DecisionTreeClassifier(random_state=0, max_depth=10)
#from sklearn.ensemble import GradientBoostingClassifier

#clf=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
#from sklearn.neighbors import KNeighborsClassifier

#clf = KNeighborsClassifier(n_neighbors=3)
#from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB

#from sklearn.ensemble import GradientBoostingClassifier

#clf=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100)
from sklearn.linear_model import LogisticRegression

clf= LogisticRegression(random_state=0)
#clf = RandomForestClassifier()

#clf=KNeighborsClassifier()

#clf=GaussianNB()

clf.fit(train_data, target)

p = clf.predict(test_data)

print(p)

d=[]

for i in range(0,418):

    if p[i]==1:

        d.append(i+2)

print(d)        

print(len(d))

re = clf.score(train_data, target)

print(re)
#TN,FP,FN,TP=confusion_matrix(Y_test,clf.predict(X_test)).ravel()

#a=(TP+TN)/(TP+TN+FP+FN)

#print('Test accuracy',a)
#y_pred=clf.predict(test_data)

#print(y_pred.shape)

#print(y_pred)





    
output = pd.DataFrame({'PassengerId':q, 'Survived': p})

output.to_csv('my_submission.csv', index=False)

output.groupby('Survived').count()