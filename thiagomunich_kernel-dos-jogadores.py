# Grupo composto pelos alunos:

#    Thiago Munich - 3628

#    Thalles Caltabiano - 3528

#    Pedro Henrique Cunha - 2727

#    Douglas Boaventura - 5144 

#    Yann Vitor - 4797 
# Importando as bibliotecas necessárias

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.cluster import KMeans

import seaborn as sb

%matplotlib inline
#  Armazenando em 'path' o arquivo dados.csv

path = '../input/dados.csv'

# Criando um dataframe

df = pd.read_csv(path)


# Excluindo as colunas que não serão usadas

cols = 'month day_of_week pdays poutcome campaign previous'.split()

df.drop(columns=cols,axis=1,inplace=True)



# Como o dataframe conta com 4 estados civis (married, single, divorced, unknown),

# se torna bem plausível juntar (single,divorced,unknown) em apenas 'single'.

# Sendo assim, a coluna 'marital' agora tem apenas dois valores (married,single)

df['marital'].replace(to_replace=['divorced', 'unknown'],value='single',inplace=True)



# Alterando os valores 'unknown' na coluna 'housing' para o valor 'no'

df['housing'].replace(to_replace=['unknown'],value='no',inplace=True)
# Visualizando os 5 primeiros registros do dataframe, após o tratamento inicial

df.head()
# Visualizando a matriz de correlação entre as colunas

plt.figure(figsize=(12,6))

sb.heatmap(df.corr(),annot=True,cmap='magma',fmt='.2f');
# Plot de indicadores economicos

plt.figure(figsize=(12,8))

plt.scatter(df['nr.employed'],df['cons.conf.idx'],alpha=0.3)

plt.xlabel('Número de funcionários')

plt.ylabel('Confiança do consumidor');
# Outro plot de indicadores economicos

plt.figure(figsize=(12,8))

plt.scatter(df['cons.price.idx'],df['cons.conf.idx'],alpha=0.3)

plt.xlabel('Índice de preço do consumidor')

plt.ylabel('Confiança do consumidor');
# Plot relativo a duração das ligações para os clientes, neste caso, em minutos

plt.figure(figsize=(12,8))

plt.plot(df['duration']/60)

plt.xlabel('Clientes')

plt.ylabel('Duração da chamada em minutos');
# Plot que mostra a idade de cada indivíduo e sua respectiva função

plt.figure(figsize=(12,8))

plt.scatter(df['age'],df['job'],alpha=0.3);

plt.xlabel('Idade do cliente')

plt.ylabel('Função exercida');
# Transformando os atributos categóricos em atributos numéricos

encoder = LabelEncoder()

df['job'] = encoder.fit_transform(df['job'])

df['marital'] = encoder.fit_transform(df['marital'])

df['education'] = encoder.fit_transform(df['education'])

df['housing'] = encoder.fit_transform(df['housing'])
# Criando o objeto 'scaler' para normalizar os dados

scaler = StandardScaler()



# 1 array vazio para armazenar as características

X = np.array([])



# Selecionando 4 colunas para serem as características do agrupamento

X = df[['age','job','marital','education','housing']]



# Aplicando a transformada normal de características

X = scaler.fit_transform(X)

# Definindo o objeto KMeans e realizando o treinamento

kmeans = KMeans(n_clusters=2,random_state=17)

model = kmeans.fit(X)
# Printando os resultados

print(model.labels_)