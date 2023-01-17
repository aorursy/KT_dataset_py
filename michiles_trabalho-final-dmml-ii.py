# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import seaborn as sns# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')

df.head()
df.shape, df.info()
# Estatísticas Descritivas

df.describe(include='all')
#Utilizando o pandas profiling para auxiliar a EDA

import pandas_profiling as pp

pp.ProfileReport(df)
# avaliação das variáveis numéricas por meio de histogramas

import matplotlib.pyplot as plt

%matplotlib inline

df.hist(figsize=(20,10))
MissingValues =df.isnull().sum().rename_axis('Colunas').reset_index(name='Missing Values')

MissingValues
# retirando os na

df2 = df.copy()

df2.dropna(axis=0,how='any',inplace= True)

df2.info(), df2.isna().any() 

import matplotlib.pyplot as plt

%matplotlib inline

df2.hist(figsize=(25,14),bins=10)
# Correlação das variáveis numéricas

plt.figure(figsize= (15, 15))



sns.heatmap(df2.corr(), square=True, annot=True, linewidth=0.5)
dfWithBin = df.copy()

bins=[0,3,15] 

group=['Low','High'] 

dfWithBin['DELINQ_bin']=pd.cut(dfWithBin['DELINQ'],bins,labels=group)

LOAN_bin=pd.crosstab(dfWithBin['DELINQ_bin'],dfWithBin['BAD'])

LOAN_bin.div(LOAN_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,title='Cruzamento de Linhas de Crédito Inadimplentes e Maus pagadores')

plt.xlabel('DELINQ')

P= plt.ylabel('%')
#avaliacao dos default loans



df2[df2['BAD']==1].drop('BAD', axis=1).describe().style.format("{:.2f}")
# Avaliando as variáveis categóricas em relacao ao pefil do pagador



JOB=pd.crosstab(df['JOB'],df['BAD'])

JOB.div(JOB.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='Tipos de Empregos e Clientes', figsize=(4,4))
REASON=pd.crosstab(df['REASON'],df['BAD'])

REASON.div(REASON.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='Tipos de Empregos e Razões', figsize=(4,4))
# Gerando Dummies para modelos que utilizam apenas variaveis numéricas



df2 = pd.get_dummies(df2, columns=['REASON', 'JOB'])
df2.head().T
#Normalizando os dados para facilitar possível visualizacoes



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df3 = pd.DataFrame(sc.fit_transform(df2), columns=df2.columns)
# importando a biblioteca

from sklearn.model_selection import train_test_split
#Etapa 1- Primeiro Separando em Treino e Teste, parâmetro test_size = 0.25 (default)

treino, teste = train_test_split(df2, random_state=42)



#Etapa 2 -  Separando o Treino em treino e validacao, para refinar o modelo

#treino, validacao = train_test_split(treino, random_state=42)



treino.shape, teste.shape # validacao.shape, 
teste.describe()
# Verificando se as amostras possuem similaridade, avaliando se há discrepância alta considerando a média e desvio padrão de cada uma. Pela análise verifica-se que a amostra gerada 

# possuem estatísticas próximas, portanto atendem ao requisito.

treino.describe()
#Selecionando as colunas que usaremos para treinar o modelo

nao_usadas = ['BAD']



# Lista das colunas que serão usadas

usadas = [c for c in treino.columns if c not in nao_usadas]
# Avaliando desempenho do modelo

#importando métrica

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

results = pd.DataFrame(columns=['Modelo', 'Accuracy', 'F1score'])
# importanto o modelo

from sklearn.ensemble import RandomForestClassifier



#instanciando o modelo

rf = RandomForestClassifier(n_estimators=200,random_state=42)
# treinando o modelo

rf.fit(treino[usadas], treino['BAD'])



#Prevendo os dados de validacao



# gerando predicoes do modelo com os dados de teste

pred_teste = rf.predict(teste[usadas])



#Medindo a acuracia nos dados de teste

results.loc[0]= ['RandonForest sem ajuste', accuracy_score(teste['BAD'],pred_teste), f1_score(teste['BAD'],pred_teste)]



accuracy_score(teste['BAD'],pred_teste), f1_score(teste['BAD'],pred_teste)

# Avaliando a importancia de cada coluna (cada variável de entrada)

pd.Series(rf.feature_importances_, index=usadas).sort_values().plot.barh()
# importando a bilbioteca para plotar o gráfico de Matriz de Confusão

import scikitplot as skplt



# Matriz de Confusão - Dados de Validação

skplt.metrics.plot_confusion_matrix(teste['BAD'], pred_teste)
# Setando parametros

rf2 = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=-1, n_estimators=900,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,  class_weight='balanced')

# treinando o modelo RF2

rf2.fit(treino[usadas], treino['BAD'])
#relizando a predicao do RF2 com base teste

pred_teste2 = rf2.predict(teste[usadas])



#métrica para RF2 validacao

results.loc[1]= ['RandonForest COM ajuste', accuracy_score(teste['BAD'],pred_teste2), f1_score(teste['BAD'],pred_teste2)]



accuracy_score(teste['BAD'],pred_teste2), f1_score(teste['BAD'],pred_teste2)
# Matriz de Confusão - Dados de Validação

skplt.metrics.plot_confusion_matrix(teste['BAD'], pred_teste2)
# Importar o modelo

from xgboost import XGBClassifier



# Instanciar o modelo

xgb = XGBClassifier(n_jobs=-1, random_state=42)



# treinando o modelo

xgb.fit(treino[usadas],treino['BAD']) 



# Fazendo predições

#pred_xgb_validacao = xgb.predict(validacao[usadas])



# Metrícas XGB validacao

#accuracy_score(validacao['BAD'],pred_xgb_validacao), balanced_accuracy_score(validacao['BAD'],pred_xgb_validacao), f1_score(validacao['BAD'],pred_xgb_validacao)
# Fazendo predições

pred_xgb_teste = xgb.predict(teste[usadas])



# Metrícas XGB teste

results.loc[2]= ['XGBoost', accuracy_score(teste['BAD'],pred_xgb_teste), f1_score(teste['BAD'],pred_xgb_teste)]



accuracy_score(teste['BAD'],pred_xgb_teste), f1_score(teste['BAD'],pred_xgb_teste)
# Matriz de Confusão - Dados de Validação

skplt.metrics.plot_confusion_matrix(teste['BAD'], pred_xgb_teste)
# Importação bibliotecas

# Importação GridSearchCV.

from sklearn.model_selection import GridSearchCV



# Uso do constructor do XGBoost para criar um classifier.

xgb2 = XGBClassifier(n_jobs=-1) # Sem nada dentro, pois vamos "variar" os parâmetros.
# Para o balaceamento do gridSearchCV foram realizadas três rodadas, a partir dos best score de cada época. 

parametros = {'n_estimators':[100,500, 900, 1100],

              'learning_rate':[0.02,0.08,0.09,1.5]}
# Importando o Make Scorer

from sklearn.metrics import make_scorer



# Importando os módulos de cálculo de métricas

from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
# Criando um dicionário com as métricas que desejo calcular.

meus_scores = {'accuracy' :make_scorer(accuracy_score),

               'recall'   :make_scorer(recall_score),

               'precision':make_scorer(precision_score),

               'f1'       :make_scorer(f1_score)}



# Exemplo para o uso scoring igual ao meus_scores.

grid = GridSearchCV(estimator = xgb2,

                      param_grid = parametros,

                      cv = 10,

                      scoring = meus_scores,   # É o meus_scores

                      refit = 'f1')            # Observe que foi configurado para f1



# Imprime o melhor score(f1) e melhor parâmetro 

grid.fit(treino[usadas],treino['BAD'])
grid.best_score_, grid.best_params_
#  Caso queira dar uma olhada nos outros scores

pd.DataFrame(grid.cv_results_).sort_values('rank_test_f1')[:3].T
# Criando um objeto que os melhores parametros.

xgb_gs = grid.best_estimator_



# Visualizar o objeto para conferir os parametros.

xgb_gs
#primeira epoca 

# Fazendo predições teste

pred_xgb_gs_teste = xgb_gs.predict(teste[usadas])



# Metrícas XGB teste

results.loc[3]= ['XGBoost com GridSearchCV',accuracy_score(teste['BAD'],pred_xgb_gs_teste), f1_score(teste['BAD'],pred_xgb_gs_teste)]



accuracy_score(teste['BAD'],pred_xgb_gs_teste), f1_score(teste['BAD'],pred_xgb_gs_teste)
# Matriz de Confusão - Dados de Validação

skplt.metrics.plot_confusion_matrix(teste['BAD'], pred_xgb_gs_teste)
#instanciando o modelo

rf2=RandomForestClassifier(n_jobs=-1)



#setando parametros para o gridSearchCV

param_dict = { 'n_estimators':[100,400,800,1000],

               'criterion': ['gini','entropy']

              }



grid2 = GridSearchCV(rf2, param_dict, cv=10)



#treinando modelo

grid2.fit(treino[usadas], treino['BAD'])
#Resultados

grid2.best_params_ , grid2.best_score_
# Criando um objeto que os melhores parametros.

rf2_gs2 = grid2.best_estimator_



# Visualizar o objeto para conferir os parametros.

rf2_gs2

# predicao teste

pred_rf2_gs2_teste = rf2_gs2.predict(teste[usadas])



# metricas predicao teste

results.loc[4]= ['RandomForest com GridSearchCV', accuracy_score(teste['BAD'],pred_rf2_gs2_teste),f1_score(teste['BAD'],pred_rf2_gs2_teste)]



accuracy_score(teste['BAD'],pred_rf2_gs2_teste), f1_score(teste['BAD'],pred_rf2_gs2_teste)
# Matriz de Confusão - Dados de Validação

skplt.metrics.plot_confusion_matrix(teste['BAD'], pred_rf2_gs2_teste)
results