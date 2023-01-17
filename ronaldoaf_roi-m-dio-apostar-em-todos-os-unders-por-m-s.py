import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Carrega o CSV e seleciona apenas a data e o pl_u

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df=pd.read_csv('../input/goalline.csv', parse_dates=['data_inicio'],date_parser=dateparse,  usecols=['data_inicio', 'pl_u'] )



#Exibe as 5 primeiras linhas do DataFrame

df.head(5)
#Cria uma lista com DataFrame para cada mês do ano

dfs_mes=[ df[ df['data_inicio'].map(lambda x: x.month) == i+1 ]  for i in range(12)    ]



#Print nome dos campos

print('{:<6} {:<10} {:<10}'.format('mês', 'média', 'desvio_padrão') )



#Para cada mês do ano

for i in range(12):

    ROIs_medios=[]

    #Faz 10 mil simulações

    for _ in range(10000):

        #Selecionando um amostra aleatório de 5000 jogos e calcula o o ROI médio

        ROIs_medios+=[ dfs_mes[i].sample(5000).pl_u.mean()]

    

    #Exibe o mês, a  média e o desvio padrão

    ROIs_medios=np.array(ROIs_medios)

    print('{:<6} {:<10.2%} {:<10.2%}'.format(i+1, ROIs_medios.mean(), ROIs_medios.std()) )
import seaborn as sns



#Vamos visualiar o histograma para para o mês de Dezembro (o último processado)

sns.distplot(ROIs_medios)