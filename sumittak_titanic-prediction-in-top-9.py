import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('train.csv')

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df.head()
df.isnull().sum()
df['Age'].fillna(df['Age'].median() ,inplace = True)
df.isnull().sum()
df['Embarked'].fillna('S' , inplace = True)
df.isnull().sum()
df['Title'].value_counts()
df.isnull().sum()
df.info()

df.head()
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['Cabin'] = df['Cabin'].str[0]
df.head(5)

df['Cabin'] = np.where((df.Pclass==1) & (df.Cabin=='U'),'C',
                        np.where((df.Pclass==2) & (df.Cabin=='U'),'D',
                        np.where((df.Pclass==3) & (df.Cabin=='U'),'G',
                        np.where(df.Cabin=='T','C',df.Cabin))))
df.head()
df['Cabin'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Cabin'] = le.fit_transform(df['Cabin'])

df.head()

df = df.drop('Ticket' , axis = 1)

df = df.drop('Name' , axis = 1)


df = df.drop('PassengerId' , axis = 1)

df.head()
le = LabelEncoder()
df['Title'] = le.fit_transform(df['Title'])

le = LabelEncoder()
df['Embarked'] = le.fit_transform(df['Embarked'])
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df.head()


df.head()
df['fam_size'] = df['SibSp'] + df['Parch'] + 1
df.head()

X_train = df.drop("Survived" , axis =1)
Y_train = df['Survived']

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
df.isnull().sum()


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth= 8)
model.fit(X_train , Y_train)

t = pd.read_csv('test.csv')

t.head()
t.isnull().sum()
t['Title'] = t.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
t.head()
t['Title'] = np.where((t.Title=='Capt') | (t.Title=='Countess') | (t.Title=='Don') | (t.Title=='Dona')
                        | (t.Title=='Jonkheer') | (t.Title=='Lady') | (t.Title=='Sir') | (t.Title=='Major') | (t.Title=='Rev') | (t.Title=='Col'),'Other',t.Title)

t['Title'] = t['Title'].replace('Ms','Miss')
t['Title'] = t['Title'].replace('Mlle','Miss')
t['Title'] = t['Title'].replace('Mme','Mrs')
t.head()
t['Age'] = np.where((t.Age.isnull()) & (t.Title=='Master'),5,
                        np.where((t.Age.isnull()) & (t.Title=='Miss'),22,
                                 np.where((t.Age.isnull()) & (t.Title=='Mr'),32,
                                          np.where((t.Age.isnull()) & (t.Title=='Mrs'),37,
                                                  np.where((t.Age.isnull()) & (t.Title=='Other'),45,
                                                           np.where((t.Age.isnull()) & (t.Title=='Dr'),44,t.Age))))))  
t.isnull().sum()
t.info()
t['Fare'] = t['Fare'].fillna(t['Fare'].mean())
t['Cabin'] = t['Cabin'].fillna('Unknown')
t['Cabin'] = t['Cabin'].str[0]
t['Cabin'] = np.where((t.Pclass==1) & (t.Cabin=='U'),'C',
                        np.where((t.Pclass==2) & (t.Cabin=='U'),'D',
                        np.where((t.Pclass==3) & (t.Cabin=='U'),'G',
                        np.where(t.Cabin=='T','C',t.Cabin))))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
t['Cabin'] = le.fit_transform(t['Cabin'])
le = LabelEncoder()
t['Embarked'] = le.fit_transform(t['Embarked'])


le = LabelEncoder()
t['Sex'] = le.fit_transform(t['Sex'])

le = LabelEncoder()
t['Title'] = le.fit_transform(t['Title'])
t['fam_size'] = t['Parch'] + t['SibSp'] + 1
t.isnull().sum()
t = t.drop('PassengerId' ,axis =1)
t = t.drop('Name' ,axis =1)
t = t.drop('Ticket' ,axis =1)

t.head()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(t)
t = ss.transform(t)

t[0]

pred = model.predict(t)
pd.DataFrame(pred).to_csv('file.csv')
