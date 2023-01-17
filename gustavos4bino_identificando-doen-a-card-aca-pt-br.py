import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt





%matplotlib inline
dados = pd.read_csv('../input/heart.csv')

# dataframe.head(N) -> N indica o número de linhas que devem ser mostradas

dados.head() 
sns.distplot(dados['age'])
f, axes = plt.subplots(2, 2, figsize=(7,7), sharex=True)

sns.distplot(dados[dados['sex'] == 1]['age'], axlabel="Homens Total", ax=axes[0,0]) # Homens

sns.distplot(dados[dados['sex'] == 0]['age'], axlabel="Mulheres Total", ax=axes[0,1]) # Mulheres



# Homens e mulheres que possuem Doença Cardíaca

# dados[dados['sex'] == 1] <- Isso pode ser lido como: "dados onde dados na coluna 'sex' seja igual a 1."

# Adicionalmente, onde está essa condição, podem existir outras condições. 



homens_com_doenca_cardiaca = dados[(dados['sex'] == 1) & (dados['target'] == 1)]['age']

mulheres_com_doenca_cardiaca = dados[(dados['sex'] == 0) & (dados['target'] == 1)]['age']



sns.distplot(homens_com_doenca_cardiaca, axlabel="Homens com Doença", ax=axes[1,0]) 

sns.distplot(mulheres_com_doenca_cardiaca, axlabel="Mulheres com Doença", ax=axes[1,1]) 

sns.relplot(x='chol',y='age',hue='target', data=dados)
ax = sns.countplot('exang', hue='target', data=dados)

ax.set(xlabel='Dor no peito', ylabel='Quantidade')

legend_labels = ax.get_legend_handles_labels()[0]

ax.legend(legend_labels, ['Sem Doença', 'Com Doença'])
sns.lmplot('age', 'trestbps', hue='target', data=dados)
sns.countplot('cp', hue='target', data=dados)
ax = sns.scatterplot('age','thalach', hue='target',size='trestbps',sizes=(10,400) ,data=dados)

ax.set(xlabel='Idade', ylabel='Batimentos Máx')
sns.boxplot('target', 'oldpeak',data=dados)
sns.boxplot('target', 'slope',data=dados)
sns.countplot('target',data=dados)
dados.info()
dados.describe()
for column in dados.columns:

    print(dados[column].isna().sum(), column)
from sklearn.model_selection import train_test_split, GridSearchCV 

from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
X = dados.drop('target', axis=1)

y = dados['target']



X_treino, X_teste, y_treino, y_teste =  train_test_split(X,y)
pipeline = Pipeline([

    ('normalizer', Normalizer()),

    ('svc', SVC())

])



pipeline.fit(X_treino, y_treino)

predicted = pipeline.predict(X_teste)

print(classification_report(y_teste, predicted))
print("Score para Treino: ",pipeline.score(X_treino, y_treino))

print("Score para Teste: ", pipeline.score(X_teste, y_teste))
pipeline = Pipeline([

    ('scaler', StandardScaler()),

    ('svc', SVC())

])



pipeline.fit(X_treino, y_treino)

predicted = pipeline.predict(X_teste)

print(classification_report(y_teste, predicted))
print("Score para Treino: ",pipeline.score(X_treino, y_treino))

print("Score para Teste: ", pipeline.score(X_teste, y_teste))