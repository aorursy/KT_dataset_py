import pandas as pd

import numpy as np

import os

import seaborn as sns

import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/titanic/train.csv')
df.info()
df.describe()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
corrmat=df.corr()

plt.figure(figsize=(20, 15))

sns.heatmap(df[corrmat.index].corr(),annot=True,cmap="RdYlGn")
df.drop('Cabin',axis=1,inplace=True)
df['Ticket']
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
df.dropna(inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.info()
pd.get_dummies(df['Embarked'],drop_first=True).head()
sex = pd.get_dummies(df['Sex'],drop_first=True)

embark = pd.get_dummies(df['Embarked'],drop_first=True)
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df.head()
df = pd.concat([df,sex,embark],axis=1)
df.head()
df.drop('Survived',axis=1).head()
df['Survived'].head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), 

                                                    df['Survived'], test_size=0.1, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
logmodel.score(X_test,y_test)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y_test,predictions)

accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
test=pd.read_csv("/kaggle/input/titanic/test.csv")
test
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test.drop("Cabin",axis=1,inplace=True)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test['Fare'].fillna(test['Fare'].median(),inplace=True)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
pd.get_dummies(test['Embarked'],drop_first=True).head()

sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test = pd.concat([test,sex,embark],axis=1)

test.head()
pred = logmodel.predict(test)
pred
submit=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
submit["Survived"]=pred
# submit.to_csv("/kaggle/input/titanic/gender_submission.csv",index=False)