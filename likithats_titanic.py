import numpy as np 

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test =  pd.read_csv("/kaggle/input/titanic/test.csv")
df_train.head()
df_train.isnull().sum()
sns.countplot(x='Survived', data=df_train)
sns.countplot(x='Survived', hue='Pclass', data=df_train)
sns.distplot(df_train['Age'],kde=False, bins=int(80/5), color ='blue',hist_kws={'edgecolor':'black'})
sns.countplot(x="Survived",hue="Sex", data=df_train)
sns.countplot(x='Pclass',hue='Sex', data=df_train)
df_train['Sex'].value_counts()
gender=['male', 'female']



for gen in gender:

    

    subplot = df_train[df_train['Sex']==gen]['Age']

    

    sns.distplot(subplot, kde=False, kde_kws = {'linewidth': 3},

                 label = gen,bins=15)



plt.legend()
sns.countplot(x="Parch", hue="Survived", data=df_train)
sns.countplot(x="SibSp",hue="Survived", data=df_train)
df_train[["Sex","Survived"]].groupby("Sex", as_index=False).mean()
df_train[["Pclass", "Survived"]].groupby("Pclass", as_index=False).mean()
df_train_len = len(df_train)
df = pd.concat([df_train, df_test], axis=0, sort=False).reset_index(drop=True)

df.drop('Cabin', axis=1, inplace=True)
plt.figure(figsize=(12,8))

sns.set_style("whitegrid")

sns.boxplot(x="Pclass",y='Age', hue='Sex',data=df)

plt.yticks(np.arange(0, 80, 5))

plt.show() 
def fill_age(cols):

    

    age = cols[0]

    Pcls= cols[1]

    sex = cols[2]

    

    if pd.isnull(age):

        if Pcls==1:

            if sex=='male':

                return 42

            else:

                return 36

    

        if Pcls==2:

            if sex=='male':

                return 29

            else:

                return 28

        if Pcls==3:

            if sex=='male':

                return 25

            else:

                return 22

    else:

        return age
df['Age'] = df[['Age','Pclass', 'Sex']].apply(fill_age,axis=1)
df.isnull().sum()
df[df["Embarked"].isnull()]
df[df["Fare"].isnull()]
plt.figure(figsize=(10,8))

sns.boxplot(y="Fare", x="Embarked", data=df)
df["Embarked"] = df["Embarked"].fillna("C")
df["Fare"] = df["Fare"].fillna(np.mean(df[df['Embarked']=="C"]["Fare"]))
df.isnull().sum()
df['Title'] = [name.split('.')[0].split(',')[-1].strip() for name in df['Name']]

df['Title'].value_counts()
df['Title'] = df['Title'].replace(['Dr', 'Mme','Jonkheer','Don','Dona','Lady','the Countess','Capt'],'other')

df['Title'] = df['Title'].replace(['Ms','Mlle'],'Miss')

df['Title'] = df['Title'].replace(['Major','Rev','Col','Sir'],'Mr')

df['Title'].value_counts()
sns.heatmap(df.corr(),annot=True)
sex= pd.get_dummies(df['Sex'],drop_first=True)

embarked=pd.get_dummies(df['Embarked'],drop_first=True,prefix='Embark')

title = pd.get_dummies(df['Title'],prefix='Title')

df = pd.concat([df.drop(['Sex','Embarked','Title','Ticket'],axis=1),sex,embarked,title],axis=1)
df.drop(['Name'], axis=1,inplace=True)
df_train = df[:df_train_len]

df_test = df[df_train_len:]

len(df_train)
x = df_train.drop(['PassengerId','Survived'], axis=1)

y = df_train['Survived'].astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, stratify=y,test_size=0.4,random_state=102)
rf = RandomForestClassifier(random_state=101)

rf.fit(X_train,Y_train)

rf_train_accuracy = round(rf.score(X_train,Y_train) * 100,2)

rf_test_accuracy = round(rf.score(X_test,Y_test) * 100,2)

print(f"Train accuracy: %{rf_train_accuracy}")

print(f"Test accuracy: %{rf_test_accuracy}")
df_test.head()
predictions = rf.predict(df_test.iloc[:,2:])
submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':predictions})
submission.reset_index()

filename = 'Titanic_Predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)