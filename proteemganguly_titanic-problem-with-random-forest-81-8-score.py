import numpy

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

data=[train_data , test_data]

train_data

test_data
women = train_data.loc[train_data.Sex=='female']["Survived"]

rate=sum(women)/len(women)



print(rate)
men = train_data.loc[train_data.Sex =='male']["Survived"]

rate_men = sum(men)/len(men)



print(rate_men)
age_ref=pd.DataFrame(data= [train_data.groupby('Pclass')['Age'].mean()], columns = train_data['Pclass'].unique())

age_ref
def fill_age(pclass,age):

    if pd.isnull(age):

        return age_ref[pclass]

    else:

        return age

for d in data:

    d['Age']= train_data.apply(lambda x: fill_age(x['Pclass'],x['Age']),axis=1)

for d in data:

    print(d.isnull().sum())

    
def fill_fare(fare):

    if pd.isnull(fare):

        return train_data['Fare'].mean()

    else:

        return fare

    

def fill_embark(embarked):

    if pd.isnull(embarked):

        return train_data['Embarked'].mode().iloc[0]

    else:

        return embarked

for d in data:

    d['Fare']=train_data.apply(lambda x : fill_fare(x['Fare']),axis=1)

    d['Embarked']=train_data.apply(lambda x: fill_embark(x['Embarked']),axis=1)



for d in data:

    print(d.isnull().sum())

    
for d in data:

    d.drop(['Cabin'],axis=1,inplace=True)
for d in data:

    print(d.isnull().sum())
title_list=list()

for d in data:

    for title in d['Name']:

        title=title.split('.')[0].split(',')[1]

        title_list.append(title)

    d['Title']=title_list

    title_list=list()
for d in data:

    print(d['Title'].value_counts())

train_data['Title']=train_data['Title'].replace([' Dr',' Rev',' Col',' Mlle',' Major',' Sir',' Lady',' Capt',' Mme',' Ms',' Don',' the Countess',' Jonkheer'],' Others')

train_data['Title'].value_counts()

test_data['Title']=test_data['Title'].replace([' Col',' Rev',' Dr',' Dona',' Ms'], ' Others')

test_data['Title'].value_counts()
def get_size(df):

    if df['SibSp']+df['Parch']+1==1:

        return 'Single'

    if df['SibSp']+df['Parch']+1>1:

        return 'Small'

    if df['SibSp']+df['Parch']+1>4:

        return 'Big'

for d in data:

    d['FamilySize']=d.apply(get_size,axis=1)

    

for d in data:

    d['IsAlone']=1

    d['IsAlone'].loc[d['FamilySize']!='Single']=0
sex = pd.get_dummies(train_data['Sex'])

embark = pd.get_dummies(train_data['Embarked'])

title = pd.get_dummies(train_data['Title'])

Pclass = pd.get_dummies(train_data['Pclass'])

FamilySize = pd.get_dummies(train_data['FamilySize'])



sex2 = pd.get_dummies(test_data['Sex'])

embark2 = pd.get_dummies(test_data['Embarked'])

title2 = pd.get_dummies(test_data['Title'])

Pclass2 = pd.get_dummies(test_data['Pclass'])

FamilySize2 = pd.get_dummies(test_data['FamilySize'])



for d in data:

    d.drop(['Sex','Embarked','Name','Ticket','Title','FamilySize'],axis=1,inplace=True)

    

train_data = pd.concat([sex,embark,train_data,title,FamilySize],axis=1)

test_data = pd.concat([sex2,embark2,test_data,title2,FamilySize2],axis=1)
X = train_data.drop('Survived',axis=1)

y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
scaler = MinMaxScaler()



scaler.fit(X_train)



scaler.transform(X_train)

scaler.transform(X_test)

scaler.transform(test_data)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print(classification_report(y_test,y_pred))

print('\n')

print(confusion_matrix(y_test,y_pred))
predictions = model.predict(test_data)

pred_list = [int(x) for x in predictions]



test2 = pd.read_csv("/kaggle/input/titanic/test.csv")

output = pd.DataFrame({'PassengerId': test2['PassengerId'], 'Survived': pred_list})

output.to_csv('MySubmission3.csv', index=False)