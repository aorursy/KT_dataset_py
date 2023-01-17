# Importando os pacotes 

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
##Colunas que serão usadas no modelo

colUtilizadasTreino = ['Property Id',

                 'ENERGY STAR Score'

                 ,'Site EUI (kBtu/ft²)',

                 'Weather Normalized Site EUI (kBtu/ft²)',

                 'Weather Normalized Site Electricity Intensity (kWh/ft²)',

                 'Weather Normalized Site Natural Gas Intensity (therms/ft²)',

                 'Weather Normalized Source EUI (kBtu/ft²)',                 

                 'Weather Normalized Site Natural Gas Use (therms)',

                 'Electricity Use - Grid Purchase (kBtu)',

                 'Weather Normalized Site Electricity (kWh)',

                 'Total GHG Emissions (Metric Tons CO2e)',

                 'Direct GHG Emissions (Metric Tons CO2e)',

                 'Indirect GHG Emissions (Metric Tons CO2e)',

                 'Property GFA - Self-Reported (ft²)',

                 'Source EUI (kBtu/ft²)'

                ]
##Colunas que serão usadas no modelo

colUtilizadasTeste = ['Property Id',

                 'Site EUI (kBtu/ft²)',

                 'Weather Normalized Site EUI (kBtu/ft²)',

                 'Weather Normalized Site Electricity Intensity (kWh/ft²)',

                 'Weather Normalized Site Natural Gas Intensity (therms/ft²)',

                 'Weather Normalized Source EUI (kBtu/ft²)',                 

                 'Weather Normalized Site Natural Gas Use (therms)',

                 'Electricity Use - Grid Purchase (kBtu)',

                 'Weather Normalized Site Electricity (kWh)',

                 'Total GHG Emissions (Metric Tons CO2e)',

                 'Direct GHG Emissions (Metric Tons CO2e)',

                 'Indirect GHG Emissions (Metric Tons CO2e)',

                 'Property GFA - Self-Reported (ft²)',

                 'Source EUI (kBtu/ft²)'

                ]
##Limpar as colunas que não influenciam o modelo

dfTeste = pd.read_csv("../input/dataset_teste.csv", usecols = colUtilizadasTeste)

##Trocar os registros que contenham a informação de 'not Available' por 0

dfTeste.replace('Not Available', 0, inplace = True)

dfTeste
##Limpar as colunas que não influenciam o modelo

dfTreino = pd.read_csv("../input/dataset_treino.csv", usecols = colUtilizadasTreino)

##Trocar os registros que contenham a informação de 'not Available' por 0

dfTreino.replace('Not Available', 0, inplace = True)

dfTreino
#Separando features e target para criação do modelo

x = dfTreino.drop('ENERGY STAR Score',axis=1)

y = dfTreino['ENERGY STAR Score']
y
#criando o modelo

Modelo = DecisionTreeClassifier(max_depth=3,random_state=0)

Modelo.fit(x,y)
#Verificando score no conjunto de treino

Modelo.score(x,y)
#Preparar para gerar um arquivo csv para mandar para o Kaggle

submission = pd.DataFrame()

submission['PropertyId'] = dfTeste['Property Id']

submission['Score'] = Modelo.predict(dfTeste)
#gerar o arquivo

submission.to_csv('sampleSubmission.csv',index=False)
dfResultado = pd.read_csv("sampleSubmission.csv")

dfResultado