# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression
df_train = pd.read_csv("../input/train.csv")

df_test    = pd.read_csv("../input/test.csv")





df_train.head()
df_train.info()
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())

df_train['Age'] = df_train['Age'].astype(int)

df_test['Age']    = df_test['Age'].astype(int)

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']

sns.countplot(x='Survived', hue="FamilySize", data=df_train, order=[1,0])

def set_familySize(x):

    if x['FamilySize'] > 0:

        return 1

    else:

        return 0



df_train['FamilySize'] = df_train.apply(set_familySize,axis=1)

df_train['FamilySize'].unique()



df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']

df_test['FamilySize'] = df_test.apply(set_familySize,axis=1)

df_test['FamilySize'].unique()

#titanic_df['Age'].values.reshape(-1,1)

#logistic.fit(titanic_df['Age'],titanic_df['Survived'])

#titanic_df['Title']= titanic_df['Name'].map(lambda x: x.split(',')[1].split('.')[0])

#titanic_df['Title'].unique()

#Person classifier as child, male and d

def get_child(x):

    age,sex = x

    return 'child' if age < 16 else sex

df_train['Person'] = df_train[['Age','Sex']].apply(get_child,axis=1)

df_test['Person']    = df_test[['Age','Sex']].apply(get_child,axis=1)

df_train['Person'].unique()

#titanic_df.drop(['Sex'],axis=1,inplace=True)

#test_df['Person'] = titanic_df[['Age','Sex']].apply(get_child,axis=1)
#Get the females

#df_train.drop(['Sex'],axis=1,inplace=True)

#test_df.drop(['Sex'],axis=1,inplace=True)

p_dummies_train  = pd.get_dummies(df_train['Person'])

p_dummies_train.columns = ['Child','Female','Male']

p_dummies_train.drop(['Male'], axis=1, inplace=True)



df_train = df_train.join(p_dummies_train)



p_dummies_test  = pd.get_dummies(df_test['Person'])

p_dummies_test.columns = ['Child','Female','Male']

p_dummies_test.drop(['Male'], axis=1, inplace=True)

df_test    = df_test.join(p_dummies_test)
df_train.info()



df_test.info()
df_test.info() 
#df_train.drop('PassengerId',axis=1,inplace=True)

#df_test.drop('PassengerId',axis=1,inplace=True)
df_train.drop('Pclass',axis=1,inplace=True)

df_test.drop('Pclass',axis=1,inplace=True)
df_train.drop('Name',axis=1,inplace=True)

df_test.drop('Name',axis=1,inplace=True)
df_train.drop('Ticket',axis=1,inplace=True)

df_test.drop('Ticket',axis=1,inplace=True)
df_train.drop('Cabin',axis=1,inplace=True)

df_test.drop('Cabin',axis=1,inplace=True)
df_train.info()
df_train.drop('Fare',axis=1,inplace=True)

df_test.drop('Fare',axis=1,inplace=True)
df_train.drop('Embarked',axis=1,inplace=True)

df_test.drop('Embarked',axis=1,inplace=True)
df_train.drop('SibSp',axis=1,inplace=True)

df_test.drop('SibSp',axis=1,inplace=True)
df_train.drop('Parch',axis=1,inplace=True)

df_test.drop('Parch',axis=1,inplace=True)
df_train.info()

df_test.info()
df_test.drop('Sex',axis=1,inplace=True)

df_train.drop('Sex',axis=1,inplace=True)
df_test.info()
df_test.drop('Person',axis=1,inplace=True)

df_train.drop('Person',axis=1,inplace=True)

df_train.info()

df_test.info()

#X_train = titanic_df.drop('Survived',axis=1)

#Y_train = titanic_df['Survived']

#X_test = test_df

df_t = df_test['PassengerId']

#df_train.drop('PassengerId',axis=1,inplace=True)

#df_test.drop('PassengerId',axis=1,inplace=True)

df_train.info()

df_test.info()
X_train = df_train.drop('Survived',axis=1)

#X_train = df_train.drop('PassengerId',axis=1)

Y_train = df_train['Survived']

X_test = df_test

#X_test = df_test.drop('PassengerId',axis=1)



log = LogisticRegression()

log.fit(X_train,Y_train)
Y_pred = log.predict(X_test)

Y_pred
log.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId": df_t,

        "Survived": Y_pred

    })

submission.to_csv('result.csv', index=False)