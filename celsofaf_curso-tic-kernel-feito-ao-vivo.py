# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/train.csv')

df_test = pd.read_csv('/kaggle/input/test.csv')
df_train.head()
df_test.head()
dados = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)
dados.head()
dados.tail()
dados.info()
# Embarked

dados['Embarked'].value_counts()
dados['Embarked'].fillna('S', inplace=True)
dados[dados['Fare'].isnull()]
media = dados[dados['Pclass'] == 3]['Fare'].mean()

dados['Fare'].fillna(media, inplace=True)
media
for pc in sorted(dados['Pclass'].unique()):

    for sex in dados['Sex'].unique():

        loc = (dados['Pclass'] == pc) & (dados['Sex'] == sex)

        media = dados[loc]['Age'].mean()

        nulos = dados[loc]['Age'].isnull().sum()

        conhecidos = len(dados[loc]['Age']) - nulos

        print('Classe {}, gênero {} --> {:.1f}'.format(pc, sex, media))

        print('Idade conhecida: {}, desconhecida: {}'.format(conhecidos, nulos))

        dados.loc[loc, 'Age'] = dados[loc]['Age'].fillna(media)

dados['Age'].describe()
dados.info()
dados['Cabin'].value_counts()
dados['Cabin'].fillna('X', inplace=True)

dados['Cabin'] = dados['Cabin'].apply(lambda s: s[0])
dados['Cabin'].value_counts()
dados.info()
dados.describe()
dados.drop('Ticket', axis=1, inplace=True)
novas_colunas_pclass = pd.get_dummies(dados['Pclass'], prefix='class') 

novas_colunas_sex = pd.get_dummies(dados['Sex'], prefix='sex') 

novas_colunas_embarked = pd.get_dummies(dados['Embarked'], prefix='embarked') 

novas_colunas_embarked = pd.get_dummies(dados['Cabin'], prefix='cabin') 



dados = pd.concat([dados, novas_colunas_pclass, novas_colunas_sex, novas_colunas_embarked, novas_colunas_embarked], axis=1)

dados.drop(['Pclass', 'Sex', 'Embarked', 'Cabin'], axis=1, inplace=True)
dados.info()
dados['Family'] = dados['SibSp'] + dados['Parch']

dados.drop(['SibSp', 'Parch'], axis=1, inplace=True)
train = dados.loc[:len(df_train)-1]

test = dados.loc[len(df_train):]
train = train.drop(['PassengerId', 'Name'], axis=1)

test = test.drop(['Survived', 'PassengerId', 'Name'], axis=1)
train.head()
test.head()
y_train = train['Survived'].values

train.drop('Survived', axis=1, inplace=True)
X_train, X_test = train.values, test.values
from sklearn.model_selection import KFold

np.random.seed(5)
kf = KFold(n_splits=5, shuffle=True, random_state=5)
#Função idêntica à usada nos modelos de regressão.

def avalia_classificador(clf, kf, X, y, f_metrica):

    metrica_val = []

    metrica_train = []

    for train, valid in kf.split(X,y):

        x_train = X[train]

        y_train = y[train]

        x_valid = X[valid]

        y_valid = y[valid]

        clf.fit(x_train, y_train)

        y_pred_val = clf.predict(x_valid)

        y_pred_train = clf.predict(x_train)

        metrica_val.append(f_metrica(y_valid, y_pred_val))

        metrica_train.append(f_metrica(y_train, y_pred_train))

    return np.array(metrica_val).mean(), np.array(metrica_train).mean()
def apresenta_metrica(nome_metrica, metrica_val, metrica_train, percentual = False):

    c = 100.0 if percentual else 1.0

    print('{} (validação): {}{}'.format(nome_metrica, metrica_val * c, '%' if percentual else ''))

    print('{} (treino): {}{}'.format(nome_metrica, metrica_train * c, '%' if percentual else ''))
from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.metrics import accuracy_score, roc_auc_score
lr = LogisticRegression(solver='liblinear')

media_acuracia_val, media_acuracia_train = avalia_classificador(lr, kf, X_train, y_train, accuracy_score) 

apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

media_auc_val, media_auc_train = avalia_classificador(lr, kf, X_train, y_train, roc_auc_score) 

apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)
dt = tree.DecisionTreeClassifier(max_depth=3)

media_acuracia_val, media_acuracia_train = avalia_classificador(dt, kf, X_train, y_train, accuracy_score) 

apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

media_auc_val, media_auc_train = avalia_classificador(dt, kf, X_train, y_train, roc_auc_score) 

apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)
preds = dt.fit(X_train, y_train).predict(X_test)
resultado = {'PassengerId': df_test['PassengerId'], 'Survived': preds.astype('int')}

resultado = pd.DataFrame.from_dict(resultado)
resultado.head()
resultado.to_csv('resultado.csv', index=False, header=True)