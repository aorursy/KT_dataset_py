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
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# classificador
from sklearn.linear_model import SGDClassifier

# selecao de dados de treino e teste
from sklearn.model_selection import train_test_split

# exibir metricas
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# vetorizador
from sklearn.feature_extraction.text import TfidfVectorizer
print('reading dataset')
base = pd.read_csv('../input/dataset-lyrics-music-mini/dataset-lyrics-musics-mini.csv')
def eda(dataset, title='EDA'):
    print(f'=={title}==')
    print('INFO \n')
    print('\nHEAD \n', dataset.head())
    print('\nTAIL \n', dataset.tail())
    print('\nDESCRIBE \n', dataset.describe())
    print('\n5 SAMPLES \n', dataset.sample(5))
    print('\nNULLS QTY \n', dataset.isnull().sum().sum())
    print('\nSHAPE \n', dataset.shape)
eda(base)
print('setup X (feature), y (target) and nomes [singer names] variables')
X = base['letra']
y = base['cantorId']
nomes = base['cantorNome'].unique()
print(f'Cantores [singers] in this dataset:\n {nomes}')
print('converting all words to lower case')
palavras = X.str.lower().str.split()
print('creating a dictionary')
dicionario = set()
for i in palavras:
    dicionario.update(i)
minhasPalavras = dict(zip(dicionario, range(len(dicionario))))
print(len(minhasPalavras), 'palavras [words]')
for i in range(0, 10):
    print(list(minhasPalavras.items())[i])
print('splitting train and test data')
Xtreino, Xteste, ytreino, yteste = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
print('vetorizing Train Data')
txtvetorizador = TfidfVectorizer()

vetorXtreino = txtvetorizador.fit_transform(Xtreino)
# treinando
print('training')
modelo = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
modelo.fit(vetorXtreino, ytreino)
print('vetorizing Test Data')
vetorXteste = txtvetorizador.transform(Xteste)
print('predicting')
previsao = modelo.predict(vetorXteste)
print(metrics.classification_report(yteste.values, previsao, target_names=nomes))

print(nomes, modelo.classes_)
confusion_matrix = confusion_matrix(yteste.values, previsao)
print(confusion_matrix)
plt.matshow(confusion_matrix, cmap='RdBu_r')
plt.title("Matriz de confusão")
plt.colorbar()
plt.ylabel("Classificações corretas")
plt.xlabel("Classificações")
print(nomes)
pd.crosstab(yteste.values, previsao, rownames=['Real'], colnames=['Previsto'], margins=True)
# insert new lyrics snatch 
novosTrechos = [
    "we used to say we live and let live",
    "Proyecto de vida en comúnlLo sé todo el abismo que ves",
    "Inch worm, inch worm. Measuring the marigolds"
]
# create a new txt vectorized 
novoVetor = txtvetorizador.transform(novosTrechos)
# build a  predict
previsao = modelo.predict(novoVetor)
# display predicts
print('Previsões [predicts]')
for trecho, artista in zip(novosTrechos, previsao):
    print(f'Trecho [snatch ]: {trecho}')
    print(f'Artista previsto [artirst predicted]: {nomes[artista]}')