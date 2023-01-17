# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.externals import joblib

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test = pd.read_csv("../input/codenation-enem2/test.csv")

train = pd.read_csv("../input/codenation-enem2/train.csv")

df_resposta = pd.DataFrame()
test
train
print(set(test.columns).issubset(set(train.columns)))
df_resposta['NU_INSCRICAO'] = test['NU_INSCRICAO']
test = test.select_dtypes(include=['int64','float64']) # selecionar todos os campos com o tipo escolhido (float)

# train = train.select_dtypes(include=['int64','float64'])
features = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']

plt.figure(figsize=(4,4))

sns.heatmap(test[features].corr(), annot=True, linewidth=0.5, linecolor='black', cmap='Greens')

plt.xticks(rotation=90)

plt.show()
train = train.loc[(train['NU_NOTA_CN'].notnull()) & (train['NU_NOTA_CH'].notnull())  & (train['NU_NOTA_LC'].notnull()) & (train['NU_NOTA_REDACAO'].notnull()) & (train['NU_NOTA_MT'].notnull())]

# test = test.loc[(test['NU_NOTA_CN'].notnull()) & (test['NU_NOTA_CH'].notnull())  & (test['NU_NOTA_LC'].notnull()) & (test['NU_NOTA_REDACAO'].notnull())]
# test[features].isnull().sum() # Não existe mais linhas com valores nulos

test[features].notnull().sum() # Não existe mais linhas com valores nulos
# train['NU_NOTA_CN'].fillna(train['NU_NOTA_CN'].mean(), inplace=True)

# train['NU_NOTA_CH'].fillna(train['NU_NOTA_CH'].mean(), inplace=True)

# train['NU_NOTA_REDACAO'].fillna(train['NU_NOTA_REDACAO'].mean(), inplace=True)

# train['NU_NOTA_LC'].fillna(train['NU_NOTA_LC'].mean(), inplace=True)

# test['NU_NOTA_CN'].fillna(train['NU_NOTA_CN'].mean(), inplace=True)

# test['NU_NOTA_CH'].fillna(train['NU_NOTA_CH'].mean(), inplace=True)

# test['NU_NOTA_REDACAO'].fillna(train['NU_NOTA_REDACAO'].mean(), inplace=True)

# test['NU_NOTA_LC'].fillna(train['NU_NOTA_LC'].mean(), inplace=True)



train.NU_NOTA_CN.fillna(train.NU_NOTA_CN.mean(), inplace=True)

train.NU_NOTA_CH.fillna(train.NU_NOTA_CH.mean(), inplace=True)

train.NU_NOTA_REDACAO.fillna(train.NU_NOTA_REDACAO.mean(), inplace=True)

train.NU_NOTA_LC.fillna(train.NU_NOTA_LC.mean(), inplace=True)

test.NU_NOTA_CN.fillna(train.NU_NOTA_CN.mean(), inplace=True)

test.NU_NOTA_CH.fillna(train.NU_NOTA_CH.mean(), inplace=True)

test.NU_NOTA_REDACAO.fillna(train.NU_NOTA_REDACAO.mean(), inplace=True)

test.NU_NOTA_LC.fillna(train.NU_NOTA_LC.mean(), inplace=True)
y = train['NU_NOTA_MT']

y
# Definição do dataset de treino somente com as informações relevantes para treinar o modelo

# features = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO','NU_NOTA_MT']

features = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']

x_train = train[features]

x_train
scaler = preprocessing.StandardScaler().fit(x_train)

scaler
X_train_scaled = scaler.transform(x_train)

print('Média: {}'.format(X_train_scaled.mean(axis=1)))

print('Desvio padrao: {}'.format(X_train_scaled.std(axis=0)))
features = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']

x_test = test[features]
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=200, n_jobs=-1, warm_start=True))
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],

                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train_scaled, y)
print(clf.best_params_)
x_test
pred_notas = clf.predict(x_test)
print(pred_notas)

print('Tamanho: {}'.format(pred_notas.size))
pred_notas
np.around(pred_notas,2)
df_resposta['NU_NOTA_MT'] = pd.DataFrame(data=np.around(pred_notas,2))

df_resposta
df_resposta.groupby('NU_NOTA_MT').size()
df_resposta['NU_NOTA_MT'].unique()
df_resposta.to_csv('answer.csv', index=False, header=True)


# Salvar o modelo preditivo

joblib.dump(clf, 'rf_regressor.pkl')



# Usar/carregar o modelo preditivo

clf2 = joblib.load('rf_regressor.pkl')

clf2.predict(x_test)