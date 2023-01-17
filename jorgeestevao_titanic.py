# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='inferno')

plt.show()
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='inferno')

plt.show()
sns.set_style('darkgrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='inferno')

plt.show()
train['Age'].hist(bins=30,color='darkred',alpha=0.7)

plt.show()
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

plt.show()
x = train[train['Pclass']==1]['Age'].mean()

y = train[train['Pclass']==2]['Age'].mean()

z = train[train['Pclass']==3]['Age'].mean()

print(x,y,z)
def idade(col):

    idade = col[0]

    Pclass = col[1]

    

    if pd.isnull(idade):



        if Pclass == 1:

            return 38



        elif Pclass == 2:

            return 30



        else:

            return 25



    else:

        return idade
train['Age'] = train[['Age','Pclass']].apply(idade,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
sex = pd.get_dummies(train['Sex'],drop_first=True)  # drop_first=True > Para evitar a multi-colinaridade

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)

train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=50)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
cm = confusion_matrix(y_test, predictions)

accuracy = int(cm[0:1,0:1] + cm[1:,1:])/cm.sum()

print('Accuracy = ',(accuracy*100).round(2),'%')

cm = pd.DataFrame(confusion_matrix(y_test, predictions))

cm.columns = ['Pred No', 'Pred Yes']

cm = cm.rename(index={0: 'No', 1:'Yes'})

cm
test = pd.read_csv('/kaggle/input/titanic/test.csv')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
def idade(col):

    idade = col[0]

    Pclass = col[1]

    

    if pd.isnull(idade):



        if Pclass == 1:

            return 38



        elif Pclass == 2:

            return 30



        else:

            return 25



    else:

        return idade
test['Age'] = test[['Age','Pclass']].apply(idade,axis=1)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
del test['Cabin']
test['Sex'] = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'], drop_first=True)

test = pd.concat([test,embark],axis=1)
test.drop(['Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
test.dropna(axis=0 ,inplace=True)
test = test[['Pclass','Age','SibSp','Parch','Fare','Sex','Q','S']]
X_test
predict = logmodel.predict(test)
test['Survived'] = predict

test.head()
sns.set_style('darkgrid')

sns.countplot(x='Pclass',hue='Survived',data=train, palette='inferno')

plt.show()
sns.countplot(x='Pclass',hue='Survived',data=test ,palette='inferno')

plt.show()