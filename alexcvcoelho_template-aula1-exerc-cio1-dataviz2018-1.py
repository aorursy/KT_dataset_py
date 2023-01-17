import pandas as pd
resposta = [
                ["uf", "Qualitativa nominal"],
                ["nome_municipio", "Qualitativa nominal"], 
                ["total_eleitores", "Quantitativa discreta"], 
                ["f_16", "Quantitativa discreta"], 
                ["f_17", "Quantitativa discreta"], 
                ["f_18_20", "Quantitativa discreta"], 
                ["gen_feminino", "Quantitativa discreta"],
                ["gen_masculino", "Quantitativa discreta"]
            ] 
#variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
import pandas as pd
columns = ['uf', 'nome_municipio', 'total_eleitores', 'gen_feminino', 'gen_masculino', 'f_16', 'f_17', 'f_18_20']
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',', usecols= columns)
df.head(3)
# Frequência de UF

df['uf'].value_counts()
# Frequência de nome_municipio

df['nome_municipio'].value_counts()
# Import de bibliotecas e leitura do csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

columns = ['uf', 'nome_municipio', 'total_eleitores', 'gen_feminino', 'gen_masculino', 'f_16', 'f_17', 'f_18_20']
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',', usecols= columns)
df.head(3)
# Criação de um DF agrupado por UF

df_uf = df.groupby(['uf'], as_index=False).sum().sort_values(['total_eleitores'], ascending=False)
df_uf.head(5)
# Criação de um DF com a soma total das colunas

df_sum = df_uf.sum().to_frame().transpose()
df_sum
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.rc('legend', fontsize=14)
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)
# tamanho da figura
plt.figure(figsize=(18, 10))

# gráfico
plt.bar(
    df_uf['uf'], 
    df_uf['total_eleitores'],
    color='#00a3cc'
)

# descrições
plt.ylabel('Eleitores (Milhões)', fontsize=14)
plt.title('Quantidade de Eleitores por Estado', fontsize=20)

# estilo
plt.grid(axis='y', linestyle='-.', linewidth=0.3)

plt.show()

labels = ["Feminino ", "Masculino"]

fig, ax = plt.subplots()
ax.stackplot(df_uf['uf'], df_uf['gen_feminino'], df_uf['gen_masculino'], labels=labels)
ax.legend(loc='upper left')
plt.show()
plt.figure(figsize=(18, 10))

#dados
df_fem = df_uf.sort_values(['gen_feminino'], ascending=False)

#gráficos
plt.bar(df_fem['uf'], df_fem['gen_feminino'],   label = 'Feminino', color = '#ffb3ff')
plt.bar(df_fem['uf'], df_fem['gen_masculino'],   label = 'Masculino', color = '#80bfff')

# descrição
plt.legend()
plt.ylabel('Eleitores (Milhões)', fontsize=14)
plt.title('Eleitores por Sexo (informado) e Estado', fontsize=20)
plt.grid(axis='y', linestyle='-.', linewidth=0.3)

plt.show()
plt.figure(figsize=(10, 10))

# gráfico
plt.pie(
    [df_sum['gen_masculino'], df_sum['gen_feminino']], 
    labels=['Masculino', 'Feminino'], 
    autopct='%1.1f%%', 
    shadow=False, 
    startangle=90,
    colors=['#80bfff', '#ffb3ff']
)

# descrição
plt.title('Eleitores por Sexo', fontsize=20)

plt.show()

plt.figure(figsize=(15, 5))

# preparação dos dados
labels = ['16 anos', '17 anos', '18 a 20 anos']
eleitores = [df_sum['f_16'][0], df_sum['f_17'][0], df_sum['f_18_20'][0]]

# gráfico
plt.barh(
    labels, 
    eleitores,
    color=['#b3cce6', '#b3cce6', '#ff9999']
)

# descrição
plt.xlabel("Eleitores")
plt.title("Jovens eleitores no país", fontsize=20)

# estilo
plt.xticks(rotation='vertical')
plt.grid(axis='x', linestyle='-.', linewidth=0.3)

plt.show()