import pandas as pd



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( train_df, col='Survived')

g.map(plt.hist, 'PassengerId', bins=20)

g = sns.FacetGrid( train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid( train_df, col='Survived')

g.map(plt.hist, 'Pclass', bins=20)
#SibSp

g = sns.FacetGrid( train_df, col='Survived')

g.map(plt.hist, 'SibSp', bins=10)
#SibSp

g = sns.FacetGrid( train_df, col='Survived')

g.map(plt.hist, 'Parch', bins=10)
#SibSp

g = sns.FacetGrid( train_df, col='Survived')

g.map(plt.hist, 'Fare', bins=10)
train_df.describe()
train_df.describe(include=['O'])
train_df = train_df.drop('PassengerId', axis = 1 )

train_df.drop('Cabin',axis=1, inplace=True)

train_df.drop(['Ticket','Name'],axis=1, inplace=True)



train_df.head()
#Base de TESTE

test_df = test_df.drop("PassengerId", axis=1)

test_df = test_df.drop("Name", axis=1)

test_df = test_df.drop("Ticket", axis=1)

test_df = test_df.drop("Cabin", axis=1)



test_df.head()
train_df.isnull().sum().sort_values(ascending=False)
test_df.isnull().sum().sort_values(ascending=False)
train_df['Age'].mean()
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())

train_df.isnull().sum().sort_values(ascending=False)
#S aparece 644 vezes, então todos os valores nulos serão substituidos por S.

train_df['Embarked'].value_counts()
train_df['Embarked'] = train_df['Embarked'].fillna('S')

train_df.isnull().sum().sort_values(ascending=False)
train_df.describe(include=['O'])
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

train_df['Sex'] = labelencoder.fit_transform(train_df['Sex'])

train_df['Embarked'] = labelencoder.fit_transform(train_df['Embarked'])
train_df.head()
test_df['Sex'] = labelencoder.fit_transform(test_df['Sex'])

test_df['Embarked'] = labelencoder.fit_transform(test_df['Embarked'])

test_df.head()
classe = train_df['Survived']

atributos = train_df.drop('Survived', axis=1)

atributos.head()
classe.head()
from sklearn.model_selection import train_test_split

atributos_train, atributos_test, class_train, class_test = train_test_split(atributos, classe, test_size = 0.25 )



atributos_train.describe()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=3, random_state =0)

model = dtree.fit(atributos_train, class_train)
from sklearn.metrics import accuracy_score

classe_pred = model.predict(atributos_test)

acc = accuracy_score(class_test, classe_pred)

print("My Decision Tree acc is {}".format(acc))