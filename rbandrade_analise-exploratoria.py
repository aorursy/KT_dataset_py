# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/heart-disease-prediction-using-logistic-regression/framingham.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Fazendo o carregamento do Dataset doenças cardíacas
dados = pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
#Verificando quantidade de instâncias e variáveis
dados.shape
#Mostrando os primeiros 05 registros
dados.head()
#Mostrando graficamente a ocorrência de dados missing 
msno.matrix(dados)
#Quantificando a ocorrência de dados missing
dados.isnull().sum()
#Avaliando a distribuição de ocorrências de casos cardíacos no Dataset
dados['TenYearCHD'].value_counts()
#Plotando a ocorência de doenças cardíacas verificando uma maior prevalência de ocorrência negativa
plt.hist(dados['TenYearCHD'], bins=2)
plt.title("Contagem de casos de cardiologia")

#Plotando a correlação entre as variáveis
correlation = dados.corr(method='pearson')
correlation
#Plotando a matriz da correlação
plt.matshow(dados.corr())
plt.show()
#Usando um novo estilo para apresentar a matriz de correlação, nesta representação os valores em laranja representam as variáveis com 
#maior correlação positiva entre as variáveis indenpendentes

#Pode-se verificar também que as variáveis age,prevalentHyp,sysBP,diaBP,glucose são as que apresentar mais correlação com a variável target TenYearCHD

#Sendo possivelmente estas variáveis as que estão relacionadas aos casos de problemas cardíacos, tomando como base os dados históricos do dataset
corr = dados.corr()
corr.style.background_gradient(cmap='coolwarm')
#Montando um dataset com os dados com maior correlação
analise = dados[["age","currentSmoker","prevalentHyp","sysBP","diaBP","glucose", "TenYearCHD"]]
analise
#Verificando a ocorrência de dados missing
analise.isnull().sum()
#Calculando índice glicêmico médio do dataset e substituindo os dados missing pela média
IGM = dados["glucose"].mean()
IGM
analise.update(analise['glucose'].fillna(IGM))
#Depois de tratato o dados de glucose considerando a média nos valores faltantes, nosso dataset está pronto
analise.isnull().sum()
#Analisando a estatística dos dados
analise.describe()
#Dividindo o dataset em dados de entrada (variáveis preditoras) e dados de saída (variável target)

entradas = analise[['age','currentSmoker','prevalentHyp','sysBP','diaBP','glucose']]
saidas = analise[["TenYearCHD"]]
#Verificando as estatísticas dos dados de entrada
entradas.describe()

#Normalizando os dados de entrada já que este processo é importante para os classificadores
entradas = preprocessing.scale(entradas)
#Apresetando as variáveis preditoras normalizadas
entradas
#Realizando o reshape dos dados e separando em 80% para treinamento e 20% para teste
#analise, target = np.arange(10).reshape((5, 2)), range(5)

data_train, data_test, target_train, target_test = train_test_split(entradas,saidas, test_size=0.20, random_state=42)
#Instanciando e treinando o modelo de arvore 
clf = tree.DecisionTreeClassifier() # instância do classificador
clf.fit(data_train, target_train) # fit encontra padrões nos dados para o argoritmo de árvore de decisão 
cRF = RandomForestClassifier() # instância do classificador
cRF.fit(data_train, target_train)  #fit encontra padrões nos dados para o argoritmo Randon Forest
cXGB = xgb.XGBClassifier() # instância do classificador
cXGB.fit(data_train, target_train) #fit encontra padrões nos dados para o argoritmo xgBoost
#Realizando a predição dos dados de teste como o modelo já treinado 
previsao = clf.predict(data_test)
previsao
previsaoRF = cRF.predict(data_test)
previsaoRF
previsaoXGB = cXGB.predict(data_test)
#Mostrando a árvore de decisão
tree.plot_tree(clf) 
#Avaliando a acurácia modelo de árvore de decisão
from sklearn.metrics import accuracy_score
accuracy_score(target_test, previsao)* 100

#Avaliando o modelo de RandonForest
accuracy_score(target_test, previsaoRF)* 100

#Avaliando a acurácia do Algoritmo XGBOOST
accuracy_score(target_test, previsaoXGB)* 100
# O modelo apresenta desbalanceamento nos dados de saída, desta forma a acurácia não é a melhor métrica de avaliação
# A matriz de confusão pode ser uma saída.
dados['TenYearCHD'].value_counts()
#Imprimindo métricas do algoritmo de árvore de decisão
from sklearn.metrics import confusion_matrix
print ('Accuracy:', accuracy_score(target_test, previsao))
print ('F1 score:', f1_score(target_test, previsao,average='weighted'))
print ('Recall:', recall_score(target_test, previsao,
                              average='weighted'))
print ('Precision:', precision_score(target_test, previsao,
                                    average='weighted'))
print ('\n clasification report:\n', classification_report(target_test, previsao))
print ('\n confussion matrix:\n',confusion_matrix(target_test, previsao))

#Imprimindo métricas do algoritmo Randon Forest
from sklearn.metrics import confusion_matrix
print ('Accuracy:', accuracy_score(target_test, previsaoRF))
print ('F1 score:', f1_score(target_test, previsaoRF,average='weighted'))
print ('Recall:', recall_score(target_test, previsaoRF,
                              average='weighted'))
print ('Precision:', precision_score(target_test, previsaoRF,
                                    average='weighted'))
print ('\n clasification report:\n', classification_report(target_test, previsaoRF))
print ('\n confussion matrix:\n',confusion_matrix(target_test, previsaoRF))
#Imprimindo métricas do algoritmo do algoritmo XGBOOST
from sklearn.metrics import confusion_matrix
print ('Accuracy:', accuracy_score(target_test, previsaoXGB))
print ('F1 score:', f1_score(target_test, previsaoXGB,average='weighted'))
print ('Recall:', recall_score(target_test, previsaoXGB,
                              average='weighted'))
print ('Precision:', precision_score(target_test, previsaoXGB,
                                    average='weighted'))
print ('\n clasification report:\n', classification_report(target_test, previsaoXGB))
print ('\n confussion matrix:\n',confusion_matrix(target_test, previsaoXGB))
#Neste caso o melhor resultado foi alcançado como algoritmo XGBOOST, lembrando que para este experimento foram utitlizados todos os algoritmos 
#com hiperparâmetros default.
#Percebe-se que o resultado sofre impacto deste desbalanceamento o que caracterizamos como víes nos dados. Podemos solucinar 
#utilizando os métodos class weight, undersampling, oversamoling e SMOTE, no entanto a técnica de balanceamento será realizada em outro artigo.