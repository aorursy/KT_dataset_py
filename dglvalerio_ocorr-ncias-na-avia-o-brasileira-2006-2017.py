import pandas as pd

import seaborn as sns

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



from gensim import corpora, models



from collections import Counter



df=pd.read_csv('../input/accidents.csv',sep=',')

df.head(4)
df['ocorrencia_dia'] = pd.to_datetime(df['ocorrencia_dia'])

df['Dia'] = df['ocorrencia_dia'].map(lambda x: x.day)

df['Ano'] = df['ocorrencia_dia'].map(lambda x: x.year)

df['Mes'] = df['ocorrencia_dia'].map(lambda x: x.month)
ocorrencias_por_ano = Counter(df['Ano'])

anos = list(ocorrencias_por_ano.keys())

acidentes_ano = list(ocorrencias_por_ano.values())
ocorrencias_por_mes = Counter(df['Mes'])

meses = list(ocorrencias_por_mes.keys())

acidentes_mes = list(ocorrencias_por_mes.values())
def estacao_ano(mes):

    if mes >= 9 and mes <= 11:

        return 'Primavera'

    elif mes >= 6 and mes <= 8:

        return 'Inverno'

    elif mes >= 3 and mes <= 5:

        return 'Outono'

    else:

        return 'Verão'



df['Estação'] = df['Mes'].apply(estacao_ano)



ocorrencias_por_estacao = Counter(df['Estação'])

estacao = list(ocorrencias_por_estacao.keys())

acidentes_estacao = list(ocorrencias_por_estacao.values())
sns.set(style="whitegrid",font_scale=2)

sns.set_color_codes("dark")





fig = plt.figure(figsize=(14, 10))



sub1= fig.add_subplot(211)

sns.barplot(x=anos, y=acidentes_ano, color='g', ax=sub1)

sub1.set(ylabel="Ocorrências", xlabel="Ano", title="Ocorrências por Ano")



sub2 = fig.add_subplot(223)

sns.barplot(x=meses, y=acidentes_mes, color='b', ax=sub2)

sub2.set(ylabel="Ocorrências", xlabel="Mês", title="Ocorrências por Mês")



sub3 = fig.add_subplot(224)

sns.barplot(x=estacao, y=acidentes_estacao, color='r', ax=sub3)

texts = sub3.set(ylabel="Ocorrências", xlabel="Estação", title="Ocorrências por Estação")



plt.tight_layout(w_pad=4, h_pad=3)



tipo_incidente = Counter(df['ocorrencia_classificacao'])

incidente = sorted(tipo_incidente, key = tipo_incidente.get, reverse=True)

tipo = sorted(list(tipo_incidente.values()), reverse = True)

sns.set(style="whitegrid",font_scale=2)

sns.set_color_codes("dark")



fig = plt.figure(figsize=(14, 10))

sub3 = fig.add_subplot(111)

sns.barplot(x=incidente, y=tipo, color='b', ax=sub3)

texts = sub3.set(ylabel="Ocorrências", xlabel="Classificação da Ocorrência")

plt.tight_layout(w_pad=4, h_pad=3)
loc_list = Counter(df['ocorrencia_tipo'])

tipo2 = sorted(loc_list, key = loc_list.get, reverse = True)

events = sorted(list(loc_list.values()), reverse = True)





sns.set(style="whitegrid",font_scale=1.5)

sns.set_color_codes("dark")

fig = plt.figure(figsize=(14, 10))

sub3 = fig.add_subplot(111)

sns.barplot(x=events[:20], y=tipo2[:20], color='g', ax=sub3)

texts = sub3.set(ylabel="Ocorrência", xlabel="Número de ocorrências")

plt.tight_layout(w_pad=4, h_pad=3)
fase_count = Counter(df['aeronave_fase_voo'])

fase = sorted(fase_count, key = fase_count.get, reverse = True)

count = sorted (list(fase_count.values()), reverse = True)

sns.set(style="whitegrid",font_scale=1.5)

fig = plt.figure(figsize=(14, 10))

sub3 = fig.add_subplot(111)

sns.barplot(x=count[:20], y=fase[:20], color='b', ax=sub3)

texts = sub3.set(ylabel="Fase de operação", xlabel="Número de ocorrências")

plt.tight_layout(w_pad=4, h_pad=3)
aeronave_conta = Counter(df['aeronave_segmento_aviacao'])

if '***' in aeronave_conta: del aeronave_conta['***']

aeronave = sorted(aeronave_conta, key = aeronave_conta.get, reverse = True)

conta= sorted(list(aeronave_conta.values()), reverse = True)



sns.set(style="whitegrid",font_scale=1.5)

fig = plt.figure(figsize=(14, 10))

sub3 = fig.add_subplot(111)

sns.barplot(x=conta, y=aeronave, color='r', ax=sub3)

texts = sub3.set(ylabel="Segmento da Aviação", xlabel="Número de ocorrências")

plt.tight_layout(w_pad=4, h_pad=3)
uf_count = Counter(df['ocorrencia_uf'])



if '***' in uf_count: del uf_count['***']

uf = sorted(uf_count, key=uf_count.get, reverse=True)

count1 = sorted(list(uf_count.values()), reverse = True)



sns.set(style="whitegrid",font_scale=1.5)

fig = plt.figure(figsize=(14, 10))

sub3 = fig.add_subplot(111)

sns.barplot(x=count1, y=uf, color='b', ax=sub3)

texts = sub3.set(ylabel="Estado da Federação", xlabel="Número de ocorrências")

plt.tight_layout(w_pad=4, h_pad=3)
mortes = []

for year in anos:

    curr_data = df[df['Ano'] == year]

    mortes.append(curr_data['quantidade_fatalidades'].sum())



sns.set(style="whitegrid",font_scale=2)

sns.set_color_codes("dark")



fig = plt.figure(figsize=(14, 10))

sub3 = fig.add_subplot(111)

sns.barplot(x=anos, y=mortes, color='g', ax=sub3)

texts = sub3.set(ylabel="Fatalidades", xlabel="Ano")

plt.tight_layout(w_pad=4, h_pad=3)