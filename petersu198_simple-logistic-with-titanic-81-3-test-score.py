import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression

sns.set() #make the graphs prettier
test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



data_cleaner = [train, test]
train
sns.countplot('Survived',data=train)
plt.figure(figsize=(10,8))

sns.heatmap(train.corr(),cmap='coolwarm',annot=True)
sns.countplot('Sex',hue='Survived',data=train)
sns.countplot('Pclass',hue='Survived',data=train)
for data in data_cleaner:

    print(data.isnull().sum())

    print('\n')
for data in data_cleaner:

    plt.figure(figsize=(8,6))

    sns.heatmap(data.isnull(),cmap='viridis')
sns.boxplot(x='Pclass',y='Age',data=train)
age_ref = pd.DataFrame(data=[train.groupby('Pclass')['Age'].mean()],columns=train['Pclass'].unique())

age_ref
def fill_age(pclass,age):

    if pd.isnull(age):

        return float(age_ref[pclass])

    else:

        return age



for data in data_cleaner:

    data['Age'] = train.apply(lambda x: fill_age(x['Pclass'],x['Age']), axis=1)
def fill_fare(fare):

    if pd.isnull(fare):

        return train['Fare'].mean()

    else:

        return fare

    

def fill_embark(embark):

    if pd.isnull(embark):

        return train['Embarked'].mode().iloc[0]

    else:

        return embark

    

for data in data_cleaner:

    data['Fare'] = train.apply(lambda x: fill_fare(x['Fare']), axis=1)

    data['Embarked'] = train.apply(lambda x: fill_embark(x['Embarked']), axis=1)
for data in data_cleaner:

    data.drop(['Cabin'],axis=1,inplace=True)
for data in data_cleaner:

    print(data.isnull().sum())

    print('\n')
train
train['Name']
title_list = list()

for data in data_cleaner:

    for title in data['Name']:

        title = title.split('.')[0].split(',')[1]

        title_list.append(title)

    

    data['Title'] = title_list

    title_list = list()
for data in data_cleaner:

    print(data['Title'].value_counts())

    print('\n')
train['Title'] = train['Title'].replace([ ' Don', ' Rev', ' Dr', ' Mme',' Ms', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',

       ' the Countess', ' Jonkheer'], 'Others')

train['Title'].value_counts()
test['Title'] = test['Title'].replace([ ' Don', ' Rev', ' Dr', ' Mme',' Ms', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',

       ' the Countess', ' Jonkheer',' Dona'], 'Others')

test['Title'].value_counts()
sns.catplot(x="SibSp",kind="count", data=train, height=4.7, aspect=2.45)
sns.catplot(x="SibSp", y="Survived", kind="bar", data=train, height=4, aspect=3).set_ylabels("Survival Probability")
sns.catplot(x="Parch", y='Survived',kind="bar", data=train, height=4.5, aspect=2.5)
def get_size(df):

    if df['SibSp'] + df['Parch'] + 1 == 1:

        return 'Single'

    if df['SibSp'] + df['Parch'] + 1 > 1:

        return 'Small'

    if df['SibSp'] + df['Parch'] + 1 > 4:

        return 'Big'

    

for data in data_cleaner:

    data['FamilySize'] = data.apply(get_size,axis=1)



for data in data_cleaner:

    data['IsAlone'] = 1 

    data['IsAlone'].loc[data['FamilySize'] != 'Single'] = 0
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

title = pd.get_dummies(train['Title'],drop_first=True)

Pclass = pd.get_dummies(train['Pclass'],drop_first=True)

FamilySize = pd.get_dummies(train['FamilySize'],drop_first=True)



sex2 = pd.get_dummies(test['Sex'],drop_first=True)

embark2 = pd.get_dummies(test['Embarked'],drop_first=True)

title2 = pd.get_dummies(test['Title'],drop_first=True)

Pclass2 = pd.get_dummies(test['Pclass'],drop_first=True)

FamilySize2 = pd.get_dummies(test['FamilySize'],drop_first=True)



for data in data_cleaner:

    data.drop(['Sex','Embarked','Name','Ticket','PassengerId','Title','FamilySize'],axis=1,inplace=True)

    

train = pd.concat([sex,embark,train,title,FamilySize],axis=1)

test = pd.concat([sex2,embark2,test,title2,FamilySize2],axis=1)
X = train.drop('Survived',axis=1)

y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
scaler = MinMaxScaler()



scaler.fit(X_train)



scaler.transform(X_train)

scaler.transform(X_test)

scaler.transform(test)
logistic_model = LogisticRegression()



logistic_model.fit(X_train, y_train)



y_pred = logistic_model.predict(X_test)
print(classification_report(y_test,y_pred))

print('\n')

print(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='plasma')
predictions = logistic_model.predict(test)

pred_list = [int(x) for x in predictions]



test2 = pd.read_csv("../input/titanic/test.csv")

output = pd.DataFrame({'PassengerId': test2['PassengerId'], 'Survived': pred_list})

output.to_csv('Titanic_with_logistic.csv', index=False)