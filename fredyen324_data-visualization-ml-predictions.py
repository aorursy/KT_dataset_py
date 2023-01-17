import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train)
#Age distribution of passengers

sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
# # of siblins or spouses

sns.countplot(x='SibSp',data=train)
# Fare distribution

train['Fare'].hist(bins=100,figsize=(8,4))
train.columns
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
P_one_age = train[train['Pclass'] == 1]['Age'].mean()

P_two_age = train[train['Pclass'] == 2]['Age'].mean()

P_three_age = train[train['Pclass'] == 3]['Age'].mean()

print('Average Age in Class One : {}'.format(P_one_age))

print('Average Age in Class Two : {}'.format(P_two_age))

print('Average Age in Class Three : {}'.format(P_three_age))





T_one_age = test[test['Pclass'] == 1]['Age'].mean()

T_two_age = test[test['Pclass'] == 2]['Age'].mean()

T_three_age = test[test['Pclass'] == 3]['Age'].mean()

print('T Average Age in Class One : {}'.format(T_one_age))

print('T Average Age in Class Two : {}'.format(T_two_age))

print('T Average Age in Class Three : {}'.format(T_three_age))

def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return P_one_age



        elif Pclass == 2:

            return P_two_age



        else:

            return P_three_age



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)

test.head()
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.head()
train['Name'] = train['Name'].apply(lambda x: x.split(',')[1].split(' ')[1])

test['Name'] = test['Name'].apply(lambda x: x.split(',')[1].split(' ')[1])
train.head()
test.head()
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

Name = pd.get_dummies(train['Name'])





sex_test = pd.get_dummies(test['Sex'],drop_first=True)

embark_test = pd.get_dummies(test['Embarked'],drop_first=True)

Name_test = pd.get_dummies(test['Name'])





plt.figure(figsize=(20,5))

sns.countplot(train['Name'])



Name_feature = Name[['Mr.', 'Mrs.', 'Ms.','Miss.']]



Name_feature_test = Name_test[['Mr.', 'Mrs.', 'Ms.','Miss.']]

train.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark,Name_feature],axis=1)

test = pd.concat([test,sex_test,embark_test,Name_feature_test],axis=1)
train.head()
test.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30,random_state = 101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
logmodel_final = LogisticRegression()

logmodel_final.fit(train.drop('Survived',axis=1),train['Survived'])
test['Fare'] =  test['Fare'].replace(np.nan,test['Fare'].mean()) 

prediction_final = pd.DataFrame(logmodel_final.predict(test.drop('PassengerId',axis=1)) ,index = test['PassengerId'],columns =['Survived'])

prediction_final.index.name = 'PassengerID'
prediction_final.to_csv('Titanic_Prediction_result.csv') #Save the result
prediction_final.head()