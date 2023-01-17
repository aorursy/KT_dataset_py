#IMPORTANTO AS BIBLIOTECAS

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
#importanto o arquivo

df = pd.read_csv('/kaggle/input/fitness-data-trends/25.csv')
#coletando informações sobre a base de dados

df.info()
#abrindo a base de dados

df.head()
#estatísticas descritivas

df.describe()

#Vale ressalatar que as medianas estão bem próximas das médias, no entanto isso não se observa em variáveis categóricas como "mood", "bool_of_activie".
df.corr()
#GRÁFICO DE CORRELAÇÃO JUNTAMENTE COM A DISTRIBUIÇÃO - STEP_COUNT X CALORIES_BURNED

Correlacao= df[['step_count','calories_burned','hours_of_sleep','weight_kg', 'mood', 'bool_of_active']]

Correlacao

sns.jointplot(data=Correlacao,y='calories_burned',x='step_count',kind='reg', color='g')

#GRÁFICO DE CORRELAÇÃO JUNTAMENTE COM A DISTRIBUIÇÃO - MOOD X WEIGTH_KG

sns.jointplot(data=Correlacao,y='weight_kg',x='mood',kind='reg', color='b')
#verificando a estatística descritiva_média da contagem de passos por categoria de humor (100 = triste, 200 = neutro e 300 = feliz)

df.step_count.groupby(df.mood).describe()
#gráfico de barras que demonstra a distribuição da contagem de passos por categoria de humor (100 = triste, 200 = neutro e 300 = feliz)

sns.barplot(x = "mood", y="step_count", data = df)

plt.title('Gráfico 1. Comparação da contagem de passos por percepção do humor')

plt.legend(['100 = Triste', '200 = Neutro', '300 = Feliz'], loc='upper left', prop={'size': 7})

plt.figure(figsize=(6,5))

#gráfico violino que demonstra a distribuição da contagem de passos por categoria de humor (100 = triste, 200 = neutro e 300 = feliz)

sns.violinplot(df['mood'], df['step_count'])

plt.title('Gráfico 2. Comparação da contagem de passos por percepção do humor')

plt.legend(['100 = Triste', '200 = Neutro', '300 = Feliz'], loc='upper left', prop={'size': 7})
#verificando a estatística descritiva_média da contagem de passos por auto percepção de atividade (0 = inativo e 500 = ativo)

df.step_count.groupby(df.bool_of_active).describe()
#gráfico de barras que demonstra a distribuição da contagem de passos por categoria de auto percepção de atividade (0 =inativo e 500 =ativo)

sns.barplot(x = "bool_of_active", y="step_count", data = df)

plt.title('Gráfico 1. Comparação da contagem de passos por percepção do humor')

plt.legend(['0 = inativo', '500 = ativo'], loc='upper left', prop={'size': 7})

plt.figure(figsize=(6,5))

#gráfico violino que demonstra a distribuição da contagem de passos por categoria de auto percepção de atividade (0 =inativo e 500 =ativo)

sns.violinplot(df['bool_of_active'], df['step_count'])

plt.title('Gráfico 2. Comparação da contagem de passos por percepção do humor')

plt.legend(['0 = inativo', '500 = ativo'], loc='upper left', prop={'size': 7})
#verificando a estatística descritiva_média da CALORIAS GASTAS por categoria de humor (100 = triste, 200 = neutro e 300 = feliz)

df.calories_burned.groupby(df.mood).describe()

#gráfico de barras que demonstra a distribuição de CALORIAS GASTAS por categoria de humor (100 = triste, 200 = neutro e 300 = feliz)

sns.barplot(x = "mood", y="calories_burned", data = df)

plt.title('Gráfico 1. Comparação das calorias gastas por percepção do humor')

plt.legend(['100 = Triste', '200 = Neutro', '300 = Feliz'], loc='upper left', prop={'size': 7})

plt.figure(figsize=(6,5))

#gráfico violino que demonstra a distribuição de CALORIAS GASTAS  por categoria de humor (100 = triste, 200 = neutro e 300 = feliz)

sns.violinplot(df['mood'], df['calories_burned'])

plt.title('Gráfico 2. Comparação das calorias gastas por percepção do humor')

plt.legend(['100 = Triste', '200 = Neutro', '300 = Feliz'], loc='upper left', prop={'size': 7})
#verificando a estatística descritiva_média da contagem de passos por auto percepção de atividade (0 = inativo e 500 = ativo)

df.calories_burned.groupby(df.bool_of_active).describe()
#gráfico de barras que demonstra a distribuição de CALORIAS GASTAS por categoria de auto percepção de atividade (0 =inativo e 500 =ativo)

sns.barplot(x = "bool_of_active", y="calories_burned", data = df)

plt.title('Gráfico 1. Comparação das calorias gastas por auto percepção de atividade')

plt.legend(['0 = inativo', '500 = ativo'], loc='upper left', prop={'size': 7})

plt.figure(figsize=(6,5))

#gráfico violino que demonstra a distribuição DE CALORIAS GASTAS  por categoria de auto percepção de atividade (0 =inativo e 500 =ativo)

sns.violinplot(df['bool_of_active'], df['calories_burned'])

plt.title('Gráfico 2. Comparação das calorias gastas por auto percepção de atividade')

plt.legend(['0 = inativo', '500 = ativo'], loc='upper left', prop={'size': 7})
#verificando a estatística descritiva_média da HORAS DE SONO por categoria de humor (100 = triste, 200 = neutro e 300 = feliz)

df.hours_of_sleep.groupby(df.mood).describe()

#gráfico de barras que demonstra a distribuição de HORAS DE SONO por categoria de humor (100 = triste, 200 = neutro e 300 = feliz)

sns.barplot(x = "mood", y="hours_of_sleep", data = df)

plt.title('Gráfico 1. Comparação das horas de sono por percepção do humor')

plt.legend(['100 = Triste', '200 = Neutro', '300 = Feliz'], loc='upper left', prop={'size': 7})

plt.figure(figsize=(6,5))

#gráfico violino que demonstra a distribuição de CALORIAS GASTAS  por categoria de humor (100 = triste, 200 = neutro e 300 = feliz)

sns.violinplot(df['mood'], df['hours_of_sleep'])

plt.title('Gráfico 2. Comparação das horas de sono  por percepção do humor')

plt.legend(['100 = Triste', '200 = Neutro', '300 = Feliz'], loc='upper left', prop={'size': 7})
#verificando a estatística descritiva_média de HORAS DE SONO por auto percepção de atividade (0 = inativo e 500 = ativo)

df.hours_of_sleep.groupby(df.bool_of_active).describe()
#gráfico de barras que demonstra a distribuição de HORAS DE SONO por categoria de auto percepção de atividade (0 =inativo e 500 =ativo)

sns.barplot(x = "bool_of_active", y="hours_of_sleep", data = df)

plt.title('Gráfico 1. Comparação das horas de sono  por auto percepção de atividade')

plt.legend(['0 = inativo', '500 = ativo'], loc='upper left', prop={'size': 7})

plt.figure(figsize=(6,5))

#gráfico violino que demonstra a distribuição DE CALORIAS GASTAS  por categoria de auto percepção de atividade (0 =inativo e 500 =ativo)

sns.violinplot(df['bool_of_active'], df['hours_of_sleep'])

plt.title('Gráfico 2. Comparação das horas de sono por auto percepção de atividade')

plt.legend(['0 = inativo', '500 = ativo'], loc='upper left', prop={'size': 7})
#criando uma tabela de contigencia entre sentir-se ativo ou inativo (bool_of_activity) x humor (mood)

pd.crosstab(df. bool_of_active, df.mood, margins=True)

#criando uma tabela de contigencia entre sentir-se ativo ou inativo (bool_of_activity) x humor (mood) em percentual

pd.crosstab([df. bool_of_active], [df.mood], normalize='index', margins=True)