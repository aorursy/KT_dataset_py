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

import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df.info()
# Numeros faltantes no dataframe
total = train_df.isnull().sum().sort_values(ascending=False)
porcentagem_1 = train_df.isnull().sum() / train_df.isnull().count()*100
porcentagem_2 = (round(porcentagem_1, 1)).sort_values(ascending=False)
dados_nulos = pd.concat([total, porcentagem_2], axis=1, keys=['Total','%'])
dados_nulos.head()
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
mulheres = train_df[train_df['Sex']=='female']
homens = train_df[train_df['Sex']=='male']
ax = sns.distplot(mulheres[mulheres['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(mulheres[mulheres['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Feminino')
ax = sns.distplot(homens[homens['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(homens[homens['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Masculino')
sns.barplot(x='Sex', y='Survived', data=train_df)
sns.barplot(x='Pclass', y='Survived', data=train_df)
# salvar os índices dos datasets para recuperação posterior
train_idx = train_df.shape[0]
test_idx = test_df.shape[0]

# salvar PassengerId para submissao ao Kaggle
passengerId = test_df['PassengerId']

# extrair coluna 'Survived' e excluir ela do dataset treino
target = train_df.Survived.copy()
train_df.drop(['Survived'], axis=1, inplace=True)

# concatenar treino e teste em um único DataFrame
df_merged = pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)

print("df_merged.shape: ({} x {})".format(df_merged.shape[0], df_merged.shape[1]))
df_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_merged.head()
# age
age_median = df_merged['Age'].median()
df_merged['Age'].fillna(age_median, inplace=True)

# fare
fare_median = df_merged['Fare'].median()
df_merged['Fare'].fillna(fare_median, inplace=True)

# embarked
embarked_top = df_merged['Embarked'].value_counts()[0]
df_merged['Embarked'].fillna(embarked_top, inplace=True)
# Os valores da coluna Sex devem ser numericos 
def mudar_sexo(valor):
    if valor == 'female':
        return 1
    else:
        return 0

df_merged['Sex'] = df_merged['Sex'].map(mudar_sexo)
embarked_dummies = pd.get_dummies(df_merged['Embarked'], prefix='Embarked')
df_merged = pd.concat([df_merged, embarked_dummies], axis=1)
df_merged.drop('Embarked', axis=1, inplace=True)

display(df_merged.head())
train = df_merged.iloc[:train_idx]
test = df_merged.iloc[train_idx:]
train.shape, test.shape
random_f = RandomForestClassifier()
random_f.fit(train, target)
# verificar a acurácia do modelo
random_forest = round(random_f.score(train, target) * 100, 2)
print("Acurácia do modelo de Regressão Logística: {}".format(random_forest))
