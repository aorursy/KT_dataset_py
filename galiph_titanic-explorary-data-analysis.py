import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
titanic = pd.read_csv('../input/train.csv')
titanic[['Age']].describe()
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
sns.countplot(x='Survived',data=titanic,palette='RdBu_r');
sns.countplot(x='Survived',hue='Sex',data=titanic,palette='RdBu_r');
sns.countplot(x='Survived',hue='Pclass',data=titanic,palette='viridis');
sns.countplot(x='Pclass',hue='Survived',data=titanic,palette='viridis');
titanic[titanic.Survived == 0].hist("Age", bins=20)

plt.show()
titanic.Age = titanic.groupby('Pclass').Age.transform(lambda x: x.fillna(np.round(x.mean())))
# Extract titles from name

titanic['Title']=0

for i in titanic:

    titanic['Title']=titanic['Name'].str.extract('([A-Za-z]+)\.', expand=False)  # Use REGEX to define a search pattern
plt.figure(figsize=(15,5))

sns.barplot(x=titanic['Title'], y=titanic['Age']);

plt.show()
titanic_melted = pd.melt(titanic[['Survived', 'Age']], "Survived", var_name="Ages")
titanic_melted.head()
sns.swarmplot(x = 'Ages', y="value", hue="Survived", data=titanic_melted)

plt.show()
df = titanic.drop(['Name', 'Ticket', 'Cabin'], axis=1)
sex_conv = lambda x : 1 if x=='male' else 0 

df['Is_male'] = df['Sex'].apply(sex_conv)
df['Age'] =pd.cut(df['Age'], bins=[1, 12, 50, 200], labels=['Child','Adult','Elder'])
embarked_new = pd.get_dummies(df['Embarked'])

train = pd.concat([df[['PassengerId','Survived','Pclass','Sex', 

                       'Age', 'SibSp', 'Parch', 'Fare','Embarked','Is_male']], embarked_new], axis=1)
# Family size feature

df['FamilySize'] = df['SibSp'] + df['Parch']

df.drop('SibSp',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df.head()
titanic = pd.read_csv('../input/train.csv')

df = titanic.drop(['Ticket', 'Cabin'], axis=1)



#Converting Sex

sex_conv = lambda x : 1 if x=='male' else 0 

df['Is_male'] = df['Sex'].apply(sex_conv)



# Adding "Title" column

df['Title']=0

for i in df:

    df['Title']=df['Name'].str.extract('([A-Za-z]+)\.', expand=False)  



# Imputing Age  

df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(np.round(x.mean())))



# Converting Age

df['Age'] =pd.cut(df['Age'], bins=[1, 12, 50, 200], labels=['Child','Adult','Elder'])



# Adding Family column

df['FamilySize'] = df['SibSp'] + df['Parch']

df.drop('SibSp',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)



#Binilen limanları gruplandırma

embarked_new = pd.get_dummies(df['Embarked'])



train = pd.concat([df[['Survived','Pclass', 

                       'FamilySize','Fare',

                       'Is_male', 'Age']], embarked_new], axis=1)



X = train[['Is_male','Pclass', 'Age', "FamilySize"]]

y = train['Survived']