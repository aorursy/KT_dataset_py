import pandas as pd

import numpy as np

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



from matplotlib.ticker import PercentFormatter





%matplotlib inline

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/callcenter_marketing.csv')
dataset.shape
#Apenas para não comprometer os resultados da visualização, vamos verificar e remover os dados faltantes:

dataset.isnull().sum()
dataset.dropna(inplace=True)
dataset.isnull().sum()
plt.figure(figsize = (16,5))



sns.heatmap(dataset.corr(), annot=True, linewidths=.5, xticklabels=True, yticklabels=True)

plt.title('Correlação entre as features')
dataset.educacao.value_counts().plot.bar()

plt.title('Educação') 
plt.boxplot(dataset.idade)
Outliers = dataset[dataset.idade>70]

Inliers = dataset[dataset.idade<=70]
Inliers.resultado.value_counts()
Estado_civil = Outliers['estado_civil'].value_counts()/Outliers.shape[0]

Estado_civil.plot.bar()

plt.title('Estado Civel - Outliers') 

plt.show()



Estado_civil_Inliers = Inliers['estado_civil'].value_counts()/Inliers.shape[0]

Estado_civil_Inliers.plot.bar()

plt.title('Estado Civil - Inliers') 
pd.crosstab(dataset.educacao,dataset.estado_civil, margins=True)
Trabalho_Outliers = Outliers.profissao.value_counts().plot.bar()

plt.title('Profissão - Outliers') 

plt.show()

Trabalho_Inliers = Inliers.profissao.value_counts().plot.bar()

plt.title('Profissão - Inliers') 
Educação = dataset.educacao.unique()

data = []

for E in Educação:

    dados = dataset[dataset.educacao==E]['idade']

    data.append(dados)

sns.boxplot( data=data,palette="Set1")
resultado=dataset['resultado'].value_counts()

resultado.plot(kind='bar')

plt.title('Aceitação X Rejeição')
prof_sim=dataset['profissao'][dataset.resultado=='sim'].value_counts()

prof_nao=dataset['profissao'][dataset.resultado=='nao'].value_counts()

profissao=pd.concat([prof_sim,prof_nao],axis=1)

profissao.columns=['sim','nao']

profissao.plot.bar()