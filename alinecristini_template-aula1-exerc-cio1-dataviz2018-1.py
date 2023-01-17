import pandas as pd
resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
import numpy as np
import matplotlib.pyplot as plt
#Utilizarei somente os dados dos eleitores que não tem a obrigação de votar
resposta = [["uf", "Qualitativa Nominal"],["nome_municipio","Qualitativa Nominal"],["total_eleitores","Quantitativa Discreta"],["f_16","Quantitativa Discreta"],["f_17","Quantitativa Discreta"],["f_70_79","Quantitativa Discreta"],["f_sup_79","Quantitativa Discreta"],["gen_feminino","Quantitativa Discreta"],["gen_masculino","Quantitativa Discreta"],["gen_nao_informado","Quantitativa Discreta"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df = pd.DataFrame(df, columns=['uf','nome_municipio','total_eleitores','f_16','f_17','f_70_79','f_sup_79','gen_feminino', 'gen_masculino', 'gen_nao_informado'])
df.head(1)
#Tabela de Frequência para a coluna uf
s1 = df['uf'].value_counts()
s2 = df['uf'].value_counts(normalize=True) * 100
stats = pd.concat([s1, s2], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa'])
stats
#Tabela de Frequência para a coluna nome_municipio
s3 = df['nome_municipio'].value_counts()
s4 = df['nome_municipio'].value_counts(normalize=True) * 100
stats_mun = pd.concat([s3, s4], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa'])
stats_mun.head()
#Número de Eleitores por UF
g1 = df['uf'].value_counts()
gs1 = g1.plot.bar(title='Eleitores por UF', color='gray')
gs1.set_xlabel('Estados')
gs1.set_ylabel('Quantidade Eleitores')
gs1.plot()
#Total de Eleitores que votaram e não eram obrigados
g2 = df['f_16'].sum() + df['f_17'].sum() + df['f_70_79'].sum() + df['f_sup_79'].sum()
g2a = df['total_eleitores'].sum()

# Data to plot
labels = 'Total Eleitores', 'Eleitores Não Obrigatórios'
sizes = [g2a, g2]
colors = ['gray', 'gold']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
#Total de eleitores Fora da Faixa por sexo

g3_fem = df['gen_feminino'].sum()
g3_mas = df['gen_masculino'].sum()
g3_out = df['gen_nao_informado'].sum()

# Data to plot
labels = 'Mulheres', 'Homens', 'Não Informado'
sizes = [g3_fem, g3_mas, g3_out]
colors = ['yellowgreen', 'gold', 'lightskyblue']
explode = (0.1, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
g4 = pd.DataFrame(df, columns=['uf','gen_feminino', 'gen_masculino', 'gen_nao_informado'])
grupos = g4.groupby(by=['uf']).sum()
grupos.head()
#Total de Eleitores por sexo e estado
grupos1 = 26
fem = grupos['gen_feminino']
masc = grupos['gen_masculino']
fig, ax = plt.subplots()
indice = np.arange(grupos1)
bar_larg = 0.4
transp = 0.7
plt.bar(indice, fem, bar_larg, alpha=transp, color='coral', label='Mulheres')
plt.bar(indice + bar_larg, masc, bar_larg, alpha=transp, color='c', label='Homens')

plt.xticks(rotation='vertical')
plt.xlabel('Estados') 
plt.ylabel('Eleitores') 
plt.title('Eleitores por Estado') 
plt.xticks(indice + bar_larg, ('AC','AL','AM','AP','BA','CE','ES','GO','MA','MG','MS','MT','PA','PB','PE','PI','PR','RJ','RN', 'RO','RR','RS','SC','SE','SP','TO')) 
plt.legend() 
plt.tight_layout() 
plt.show()
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)