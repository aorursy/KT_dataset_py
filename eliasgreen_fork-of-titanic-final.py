import numpy as np # Библиотека для математических вычислений

import pandas as pd # Библиотека для математических вычислений

import seaborn as sns # Библиотека для графиков
sns.set_context('talk') # Увеличиваем шрифт для отображения графиков
# Читаем данные

df_train = pd.read_csv('/kaggle/input/titanic/train.csv') # Тренировочный набор данных - на них будет учиться наша нейронная сеть

df_test = pd.read_csv('/kaggle/input/titanic/test.csv') # Тестовый набор данных
df_train.describe() # Описываем данные
df_train.info() # Узнаем более подробную информацию о датасете
plt1 = sns.barplot(x="Pclass", y="Survived", data=df_train[['Survived', 'Pclass']])
plt2 = sns.distplot(df_train['Pclass'].dropna())
for index, row in df_train.iterrows():

    df_train.at[index,'Name'] = row['Name'].split(',')[1].split()[0]

    

df_train['Name']
plt1 = sns.barplot(x="Name", y="Survived", data=df_train[['Survived', 'Name']])

settings = plt1.set_xticklabels(plt1.get_xticklabels(),rotation=60, ha="right")
plt2 = sns.countplot(df_train['Name'].dropna())

settings = plt2.set_xticklabels(plt2.get_xticklabels(),rotation=60, ha="right")
plt1 = sns.barplot(x="Sex", y="Survived", data=df_train[['Survived', 'Sex']])
plt2 = sns.countplot(df_train['Sex'].dropna())
plt1 = sns.lineplot(x="Survived", y="Age", data=df_train[['Age', 'Survived']])
plt2 = sns.distplot(df_train['Age'].dropna())
plt1 = sns.lineplot(x="Survived", y="SibSp", data=df_train[['SibSp', 'Survived']])
plt2 = sns.countplot(df_train['SibSp'].dropna())
plt1 = sns.lineplot(x="Survived", y="Parch", data=df_train[['Parch', 'Survived']])
plt2 = sns.countplot(df_train['Parch'].dropna())
plt1 = sns.lineplot(x="Survived", y="Fare", data=df_train[['Fare', 'Survived']])
plt2 = sns.distplot(df_train['Fare'].dropna())
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['Cabin'] = df_train['Cabin'].fillna('missing')

le.fit(df_train['Cabin'])

df_train['Cabin'] = le.transform(df_train['Cabin'])



df_train['Cabin']
plt1 = sns.lineplot(x="Survived", y="Cabin", data=df_train[['Cabin', 'Survived']])
plt2 = sns.distplot(df_train['Cabin'].dropna())
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['Embarked'] = df_train['Embarked'].fillna('missing')

le.fit(df_train['Embarked'])

df_train['Embarked'] = le.transform(df_train['Embarked'])



df_train['Embarked']
plt1 = sns.lineplot(x="Survived", y="Embarked", data=df_train[['Embarked', 'Survived']])
plt2 = sns.countplot(df_train['Embarked'].dropna())
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['Name'] = df_train['Name'].fillna('missing')

le.fit(df_train['Name'])

df_train['Name'] = le.transform(df_train['Name'])



df_train['Name']
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['Sex'] = df_train['Sex'].fillna('missing')

le.fit(df_train['Sex'])

df_train['Sex'] = le.transform(df_train['Sex'])



df_train['Sex']
df_train['Age'].fillna((df_train['Age'].mean()), inplace=True)
from sklearn.neural_network import MLPClassifier

neural_network = MLPClassifier(hidden_layer_sizes=(1000, 100), activation='relu', random_state=1999)

# Нейронная сеть с 1000 нейронов на входе и 100 на внутреннем слое, активационная функция - ReLU (Rectifier)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(neural_network, df_train.drop(['Survived', 'PassengerId', 'Ticket'], axis=1), df_train['Survived'], cv=5)

print("Neural network модель")

print("Точность: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))