# Exemplo de Notebook para submissão de soluções para a Primeira Competição da TAIL.

# Autor: Joás de Brito

# Implementado e testado no ambiente de desenvolvimento da Kaggle.



# Este notebook visa guiar os iniciados em Data Science no processo de submeter uma solução para o referido campeonato.

# As células aqui presentes servem apenas como um exemplo estrutural da sua solução, ficando a cargo do competidor adicionar ou remover procedimentos.

# Para uma introdução ao ambiente de competições, visite a seção Tutorial, na página inicial deste campeonato. 

# O Setor de Projetos agradece a participação dos competidores e deseja boa sorte a cada um dos membros da TAIL.



# Desenvolvido e mantido por:

#     TAIL - Setor de Projetos
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sbn

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Aquisição do dataset



train_filepath = '/kaggle/input/tail-competicao/treino.csv'

test_filepath = '/kaggle/input/tail-competicao/teste.csv'

train_data = pd.read_csv(train_filepath)

test_data = pd.read_csv(test_filepath)



test_data['Resposta'] = pd.read_csv('/kaggle/input/tail-competicao/exemplo_submissao.csv')['Resposta']



print("Formato dos dados de treinamento: ", train_data.shape)

print("Formato dos dados de teste: ", test_data.shape)



train_data.head()

test_data.tail()
# Análise Exploratória de Dados



df = train_data



# distribuição das observações por respostas

print("\nDistribuição de respostas: ")

print(df['Resposta'].value_counts())

train_data['Resposta'].value_counts().plot.bar(color=['blue','orange'])

plt.show()



# Tipagem e estatística descritiva do dataset

print("\nInformações gerais do dataset: ")

df.info()

df.describe()



print("\nValores únicos por coluna:")

print(train_data.nunique())





# distribuição das observações por código de região

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 53))

train_data['Código_Região'].value_counts().plot.bar(color = color)

plt.title('Distribuição por Código de Região', fontsize = 15)

plt.show()



print(train_data.iloc[:12])



# visualização com base no dataset inteiro (train + test)

train_data['type'] = 'train'

test_data['type'] = 'test'

dataset = pd.concat([train_data, test_data])

sbn.distplot(dataset['Prêmio_Anual'])

plt.title('Distribuição de premiação anual', fontsize = 15)

plt.show()

# Construção e Avaliação do Modelo



# Split do dataset, one-hot encoding

X = pd.get_dummies(dataset.drop(['Resposta', 'type'], axis=1)).values

y = dataset['Resposta'].replace({"Sim": 1, "Não": 0}).values



# 5 avaliações sucessivas, para diferentes divisões entre training/testing

kfold, scores = KFold(n_splits = 5, shuffle = True, random_state = 20), list()

for train, test in kfold.split(X):

    X_train, X_test = X[train], X[test]

    y_train, y_test = y[train], y[test]

    

    # Construção do modelo

    classifier = RandomForestClassifier(n_estimators=5, random_state=0)

    classifier.fit(X_train, y_train)

    preds = classifier.predict(X_test)

    

    # Avaliação do modelo

    score = roc_auc_score(y_test, preds)

    scores.append(score)

    print("ROC AUC: ", score)



# Avaliação média do modelo

print("Average Validation ROC AUC: ", sum(scores)/len(scores))

print(preds)



# Submissão da Solução Final



X_test = pd.get_dummies(test_data.drop(['type', 'Resposta'], axis = 1)).values

preds = classifier.predict(X_test)



submission = pd.DataFrame(data = {"id": test_data['Id'].values, "Resposta": preds})

submission.to_csv('solution.csv', index = False)

submission.head()