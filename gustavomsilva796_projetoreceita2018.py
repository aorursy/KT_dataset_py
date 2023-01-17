#Importando bibliotecas



import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import locale



%matplotlib inline
#leitura do CSV



df = pd.read_csv(r'../input/r2018.csv', sep = ";")
#Exibindo o dataframe



df.head()
#Vendo as colunas e tiposd e dados



df.dtypes
#Vendo as colunas do dataframe



df.columns
df['receita_prevista'] = df['receita_prevista'].apply(lambda x: x.replace(',','.'));

df['receita_prevista_acrescimo'] = df['receita_prevista_acrescimo'].apply(lambda x: x.replace(',','.'));

df['receita_prevista_atualizada'] = df['receita_prevista_atualizada'].apply(lambda x: x.replace(',','.'));

df['receita_arrecadada'] = df['receita_arrecadada'].apply(lambda x: x.replace(',','.'));
df['receita_prevista'] = df['receita_prevista'].astype(np.float64)

df['receita_prevista_acrescimo'] = df['receita_prevista_acrescimo'].astype(np.float64)

df['receita_prevista_atualizada'] = df['receita_prevista_atualizada'].astype(np.float64)

df['receita_arrecadada'] = df['receita_arrecadada'].astype(np.float64)
#Adicionando um contador



df['count'] = 1
df.head()
#Calculando o total de receita arrecadada



valor_arrecadado = df['receita_arrecadada'].sum()

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

valor_arrecadado = locale.currency(valor_arrecadado, grouping=True, symbol=None)



print('Valor arrecadado: R${}'.format(valor_arrecadado))
#Contagem de recursos por secretaria em ordem decrescente



count_orgao = df.groupby('orgao_nome')['count'].sum().sort_values(ascending=False)

count_orgao
#Visualização em forma de gráfico de barras



count_orgao = df.groupby('orgao_nome')['count'].sum().sort_values().plot(kind='barh', grid=True, figsize=(13,13), alpha=.8)



for p in count_orgao.patches:

    b=p.get_bbox()

    count_orgao.annotate("{:,.0f}".format(b.x1 + b.x0), (p.get_x() + p.get_width() + 1, p.get_y() - 0.05))



plt.title('Quantidade de Arrecadações agrupados por Secretaria \n Total: ' + str(df['count'].sum()) + ' arrecadações')

plt.ylabel('Secretaria')

plt.xlabel('Quantidade de Arrecadações')

plt.show()
#Quantidade de recursos arrecadadas pelas fontes



count_orgao = df.groupby('fonte_recurso_nome')['count'].sum().sort_values().plot(kind='barh', grid=True, figsize=(13,13), alpha=.8)



for p in count_orgao.patches:

    b=p.get_bbox()

    count_orgao.annotate("{:,.0f}".format(b.x1 + b.x0), (p.get_x() + p.get_width() + 1, p.get_y() - 0.05))



plt.title('Quantidade de Arrecadações agrupados por Fonte \n Total: ' + str(df['count'].sum()) + ' arrecadações')

plt.ylabel('Fonte de Recurso')

plt.xlabel('Quantidade de Arrecadações')

plt.show()
#Arrecadação por Secretaria



sum_orgao_pago = df.groupby('orgao_nome')['receita_arrecadada'].sum().sort_values().plot(kind='barh', figsize=(14,14), grid=True, alpha=.8)



for p in sum_orgao_pago.patches:

    b=p.get_bbox()

    sum_orgao_pago.annotate("{:,.2f}".format(b.x1 + b.x0), (p.get_x() + p.get_width() + 1, p.get_y() - 0.05))



plt.title('Quantidade de Arrecadações agrupados por Secretaria \n Total: ' + str(df['count'].sum()) + ' arrecadações')

plt.ylabel('Secretaria')

plt.xlabel('Valor pago (em R$)')

plt.show()
#Arrecadação pela fonte de recursos



sum_orgao_pago = df.groupby('fonte_recurso_nome')['receita_arrecadada'].sum().sort_values().plot(kind='barh', figsize=(15,15), grid=True, alpha=.8)



for p in sum_orgao_pago.patches:

    b=p.get_bbox()

    sum_orgao_pago.annotate("{:,.2f}".format(b.x1 + b.x0), (p.get_x() + p.get_width() + 1, p.get_y() - 0.05))



plt.title('Quantidade de Arrecadações agrupados por Fonte de Recursos \n Total: ' + str(df['count'].sum()) + ' arrecadações')

plt.ylabel('Fonte de Recurso')

plt.xlabel('Valor pago (em R$)')

plt.show()
#Arrecadação pela natureza da receita



sum_alinea_receita = df.groupby('alinea_receita_nome')['receita_arrecadada'].sum().sort_values().plot(kind='barh', figsize=(5,15), grid=False, alpha=.9)



for p in sum_alinea_receita.patches:

    b=p.get_bbox()

    sum_alinea_receita.annotate("{:,.2f}".format(b.x1 + b.x0), (p.get_x() + p.get_width() + 1, p.get_y() - 0.05))



plt.title('Quantidade de Arrecadações agrupados pela Natureza da Receita \n Total: ' + str(df['count'].sum()) + ' arrecadações')

plt.ylabel('Natureza da receita')

plt.xlabel('Valor pago (em R$)')

plt.show()
#Quantidade de Arrecadações ao longo do ano



sum_arrecadacoes_mes=df.groupby('mes')['count'].sum().plot(figsize=(20,5), grid=True, alpha=.8, kind='barh')



for p in sum_arrecadacoes_mes.patches:

    b=p.get_bbox()

    sum_arrecadacoes_mes.annotate("{}".format(b.x1 + b.x0), (p.get_x() + p.get_width() + 1, p.get_y() - 0.05))



plt.title('Quantidade de Arrecadações ao longo do ano \n Total: ' + str(df['count'].sum()) + ' arrecadações')

plt.xlabel('Meses')

plt.ylabel('Quantidade de Arrecadações')

plt.show()
df.groupby('mes')['receita_prevista','receita_prevista_atualizada','receita_arrecadada'].sum().plot(figsize=(12,5), grid=True, kind='bar')



plt.title('Receita Prevista e Receita Arrecadada ao longo de 2018 \n Total: ' + str(df['count'].sum()) + ' arrecadações')

plt.xlabel('Meses')

plt.ylabel('Valor (em R$)')

plt.show()
print('Valor previsto: R$ {:,.2f}'.format(df['receita_prevista'].sum()))

print('Valor arrecadado: R$ {:,.2f}'.format(df['receita_arrecadada'].sum()).replace('.', ','))
perc = df['receita_arrecadada'].sum() / df['receita_prevista'].sum()

print('Valor percentual entre realizado e previsto em 2018: {:.2f}%'.format(perc*100))
dfd = pd.read_csv(r'../input/d2018.csv', sep = ";")
dfd.head()
dfd.dtypes
dfd['valor_pago'] = dfd['valor_pago'].apply(lambda x: x.replace(',','.'));
dfd['valor_pago'] = dfd['valor_pago'].astype(np.float64)
sobra_cofre = df['receita_arrecadada'].sum() - dfd['valor_pago'].sum()



print('Valor que sobrou nos cofres em 2018: R${:,.2f}'.format(sobra_cofre))