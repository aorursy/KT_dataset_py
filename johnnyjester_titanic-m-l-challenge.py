import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

        

train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
titanic_test.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Survived',hue='Sex', data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def avg_age(cols):

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
train['Age'] = train[['Age','Pclass']].apply(avg_age,axis=1)

titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(avg_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)

titanic_test.drop('Cabin',axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(titanic_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
titanic_test.fillna(0, inplace=True)
import re
def digitize(data):

    num_list = re.findall(r'\d+', data)

    if(num_list == []):

        return 3000

    num = int(''.join(map(str,num_list[-1])))

    return num
train['Ticket'] = train['Ticket'].apply(digitize)
train.head()
titanic_test['Ticket'].apply(str)

titanic_test['Ticket'] = titanic_test['Ticket'].apply(digitize)
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
sex_test = pd.get_dummies(titanic_test['Sex'],drop_first=True)

embark_test = pd.get_dummies(titanic_test['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name'],axis=1,inplace=True)
titanic_test.drop(['Sex','Embarked','Name'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
titanic_test = pd.concat([titanic_test,sex_test,embark_test],axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,  y_test= train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=101)
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

ans = logmodel.predict(titanic_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
df = pd.DataFrame(columns=['PassengerId', 'Survived'])
df['PassengerId'] = titanic_test['PassengerId']

df['Survived'] = ans
df.set_index('PassengerId').to_csv('submission.csv')