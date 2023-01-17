# standard imports & configuration

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

sns.set_style('whitegrid')
# sci-kit learn

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.metrics import confusion_matrix, classification_report
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train.head()
df_train.info()
df_train.describe()
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

passid = df_test['PassengerId']

df_test.head()
df_test.info()
df_test.describe()
plt.figure(figsize=(18,6))

sns.distplot(df_train['Age'].dropna(), bins=30)

plt.title('Titanic Age Distribution')
plt.figure(figsize=(18,6))

sns.countplot(x='SibSp', data=df_train)

plt.title('Passengers with Siblings and/or Spouses on board')
df_train['Fare'].hist(bins=40, figsize=(18,6))

plt.title('Fare')
plt.figure(figsize=(12,10))

sns.boxplot(df_train['Pclass'], df_train['Age'])

plt.title('Age vs Pclass')
plt.figure(figsize=(18,6))

sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.title('Missing Training Data')
missing_age = len([x for x in df_train['Age'].isnull() if x==True])

total_age = len([x for x in df_train['Age']])

print(f'The Age column is missing {missing_age} entries out of {total_age} total entries.')

print(f'Approximatly {round((missing_age / total_age) * 100, 2)}% of the Age column is missing data.')



missing_cabin = len([x for x in df_train['Cabin'].isnull() if x==True])

total_cabin = len([x for x in df_train['Cabin']])

print(f'The Cabin column is missing {missing_cabin} entries out of {total_cabin} total entries.')

print(f'Approximatly {round((missing_cabin / total_cabin) * 100, 2)}% of the Cabin column is missing data.')
plt.figure(figsize=(18,6))

sns.heatmap(df_test.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.title('Missing Testing Data')
missing_age = len([x for x in df_test['Age'].isnull() if x==True])

total_age = len([x for x in df_test['Age']])

print(f'The Age column is missing {missing_age} entries out of {total_age} total entries.')

print(f'Approximatly {round((missing_age / total_age) * 100, 2)}% of the Age column is missing data.')



missing_cabin = len([x for x in df_test['Cabin'].isnull() if x==True])

total_cabin = len([x for x in df_test['Cabin']])

print(f'The Cabin column is missing {missing_cabin} entries out of {total_cabin} total entries.')

print(f'Approximatly {round((missing_cabin / total_cabin) * 100, 2)}% of the Cabin column is missing data.')
def compute_age(cols):

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
df_train['Age'] = df_train[['Age', 'Pclass']].apply(compute_age, axis=1)

df_test['Age'] = df_test[['Age', 'Pclass']].apply(compute_age, axis=1)
df_train.drop('Cabin', axis=1, inplace=True)

df_test.drop('Cabin', axis=1, inplace=True)

df_train.dropna(inplace=True)

df_test.fillna(0, inplace=True)

plt.figure(figsize=(18,6))

sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.title('Missing Training Data')
plt.figure(figsize=(18,6))

sns.heatmap(df_test.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.title('Missing Testing Data')
plt.figure(figsize=(18,6))

sns.heatmap(df_train.corr().abs(), cmap='plasma')

plt.title('Correlation Heatmap')
plt.figure(figsize=(18,6))

sns.clustermap(df_train.corr().abs(), cmap='plasma')

plt.title('Cluster Heatmap')
plt.figure(figsize=(12,6))

sns.countplot(x='Survived', hue='Sex', data=df_train, palette='coolwarm')

plt.title('Survivors by Sex')
plt.figure(figsize=(12,6))

sns.countplot(x='Survived', hue='Pclass', data=df_train, palette='Set1')

plt.title('Survivors by Pclass')
men = df_train[df_train['Sex']=='male']

plt.figure(figsize=(12,6))

sns.swarmplot(x='Pclass', y='Age', hue='Survived', data=men)

plt.title('Men')
women = df_train[df_train['Sex']!='male']

plt.figure(figsize=(12,6))

sns.swarmplot(x='Pclass', y='Age', hue='Survived', data=women)

plt.title('Women')
plt.figure(figsize=(18,6))

sns.barplot(df_train['Age'], df_train['Sex'], hue=df_train['Survived'])

plt.title('Survivors by Age & Sex')
fc = sns.FacetGrid(df_train, row='Embarked', size=4, aspect=2)

fc.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='Set1')
le = LabelEncoder()
for each in df_train.columns:

    df_train[each] = le.fit_transform(df_train[each])

df_train.head()
for each in df_test.columns:

    df_test[each] = le.fit_transform(df_test[each])

df_test.head()
X_train = df_train.drop('Survived', axis=1)

y_train = df_train['Survived']

X_test = df_test
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
rfc = RandomForestClassifier(n_estimators=1000, random_state=1912)

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
y_test = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

y_test = y_test['Survived']
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
df = pd.DataFrame()

df['PassengerId'] = passid

df['Survived'] = y_pred

df.to_csv('/kaggle/working/titanic_final2.csv', index=False)
pd.set_option('display.max_rows', 500)

results = pd.read_csv('/kaggle/working/titanic_final2.csv')

#results