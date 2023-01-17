import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/BR_eleitorado_2016_municipio.csv", delimiter=";")

df.head(10)
df.shape
corr = df.corr()

plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')

corrMat = plt.matshow(corr, fignum = 1)

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.gca().xaxis.tick_bottom()

plt.colorbar(corrMat)

plt.title('Correlation Matrix')

plt.show()
df = df.drop(columns=['cod_municipio_tse'])

df.head()
x = sns.PairGrid(df)

x.map(plt.scatter)
df['uf'].value_counts().count()
uf = pd.DataFrame(df['uf'].value_counts())

plt.figure(figsize=(15,5))

sns.barplot(x=uf.index, y=uf.uf, palette='rocket')
eleitores = df[['uf','total_eleitores']].sort_values(by='uf')
plt.figure(figsize=(15,5))

plt.title("Média de eleitores por Município em cada UF")

sns.barplot(x=eleitores.uf, y=eleitores.total_eleitores)
eleitores_grpd_by_uf = eleitores.groupby(['uf']).sum()

#eleitores_grpd_by_uf.head()

plt.figure(figsize=(15,5))

plt.title("Total de eleitores em cada UF")

sns.barplot(x=eleitores_grpd_by_uf.index, y=eleitores_grpd_by_uf.total_eleitores)
norte = ['AM','RR','AP','PA','TO','RO','AC']

centroeste = ['MT','MS','GO']

sudeste = ['SP','ES','MG','RJ']

sul = ['PR','RS','SC']

nordeste = ['MA','PI','CE','RN','PE','PB','SE','AL','BA']



df_region = eleitores

df_region['regiao'] = ''



for i, r in df_region.iterrows():

    if r['uf'] in norte:

        df_region.at[i,'regiao'] = 'Norte'

    elif r['uf'] in centroeste:

        df_region.at[i,'regiao'] = 'Centro-Oeste'

    elif r['uf'] in sudeste:

        df_region.at[i,'regiao'] = 'Sudeste'

    elif r['uf'] in sul:

        df_region.at[i,'regiao'] = 'Sul'

    else:

        df_region.at[i,'regiao'] = 'Nordeste'
df_region.head()
df_ufs = pd.DataFrame(norte + centroeste + sudeste + sul + nordeste)
reg = pd.DataFrame(df_region['regiao'].value_counts())

reg.plot(kind='pie', title='Quantidade de Municípios por Região',subplots=True, figsize=(10,10))



elec = pd.DataFrame(df_region.drop(columns=['uf']).groupby(['regiao']).sum())

elec.plot(kind='pie', title='Quantidade de Eleitores por Região',subplots=True, figsize=(10,10))
plt.figure(figsize=(25,8))

sns.stripplot(x='total_eleitores', y='regiao', hue='uf', data=df_region, palette='muted', size=5, jitter=.3)
plt.figure(figsize=(10,15))

sns.violinplot(y='total_eleitores', x='regiao', data=df_region)
plt.figure(figsize=(25,8))

sns.stripplot(x='total_eleitores', y='regiao', hue='uf', data=df_region[df_region['total_eleitores'] < 100000], palette='bright', size=4, jitter=.3)
plt.figure(figsize=(10,15))

sns.violinplot(y='total_eleitores', x='regiao', data=df_region[df_region['total_eleitores'] < 100000])
gen = df[['uf','gen_feminino','gen_masculino','gen_nao_informado']]

gen.head()
gen_sum = pd.DataFrame(gen.groupby(['uf'],as_index=False).sum())

gen_sum.head()
gen_sum.set_index('uf').plot(kind='bar', stacked=True, figsize=(18,8))
gen_lst = []



for index, row in gen.iterrows():

    gen_lst.append([row['uf'],'Feminino',row['gen_feminino']])

    gen_lst.append([row['uf'],'Masculino',row['gen_masculino']])

    gen_lst.append([row['uf'],'Não Informado',row['gen_nao_informado']])

    

gen_transf = pd.DataFrame(gen_lst)

gen_transf.columns = ['uf','genero','qtd']



gen_transf.head()
gen_grpd_by_gen = pd.DataFrame(gen_transf.groupby(['uf','genero'],as_index=False).sum())

gen_grpd_by_gen.head()
sns.catplot(x="uf", y="qtd", hue="genero", data=gen_grpd_by_gen, kind="bar", palette="bright", height=8, aspect=2 )
gen_n_inf = gen[['uf','gen_nao_informado']]

gen_n_inf = gen_n_inf.groupby(['uf'],as_index=False).sum()

gen_n_inf.head()



gen_n_inf.sort_values(by='gen_nao_informado',ascending=False).plot(kind='bar', x='uf', y='gen_nao_informado', figsize=(18,8), color='g')
df_ages = df.drop(columns=['nome_municipio','gen_feminino','gen_masculino','gen_nao_informado'])

df_ages.head()
df_ages_sum = df_ages.groupby(['uf'], as_index=False).sum()

df_ages_sum.head()
df_ages_percent = []



for i, r in df_ages_sum.iterrows():

    df_ages_percent.append([r['uf'],r[2]/r[1], r[3]/r[1], r[4]/r[1], r[5]/r[1], r[6]/r[1], r[7]/r[1], r[8]/r[1], r[9]/r[1], r[10]/r[1], r[11]/r[1]])

    

df_ages_percent = pd.DataFrame(df_ages_percent)

df_ages_percent.columns = ['uf','f_16','f_17','f_18_20','f_21_24','f_25_34','f_35_44','f_45_59','f_60_69','f_70_79','f_sup_79']

df_ages_percent.iloc[:,1:] = df_ages_percent.iloc[:,1:] * 100

df_ages_percent.head()
df_ages_il = []



for i, r in df_ages_percent.iterrows():

    for x in range(1,11):

        df_ages_il.append([r['uf'], r[x]])



df_ages_il = pd.DataFrame(df_ages_il)

df_ages_il.columns = ['uf','perc']
df_ages_il = []



for i, r in df_ages_percent.iterrows():

    df_ages_il.append([r['uf'],'f_16',r['f_16']])

    df_ages_il.append([r['uf'],'f_17',r['f_17']])

    df_ages_il.append([r['uf'],'f_18_20',r['f_18_20']])

    df_ages_il.append([r['uf'],'f_21_24',r['f_21_24']])

    df_ages_il.append([r['uf'],'f_25_34',r['f_25_34']])

    df_ages_il.append([r['uf'],'f_35_44',r['f_35_44']])

    df_ages_il.append([r['uf'],'f_45_59',r['f_45_59']])

    df_ages_il.append([r['uf'],'f_60_69',r['f_60_69']])

    df_ages_il.append([r['uf'],'f_70_79',r['f_70_79']])

    df_ages_il.append([r['uf'],'f_sup_79',r['f_sup_79']])



df_ages_il = pd.DataFrame(df_ages_il)

df_ages_il.columns = ['uf','age','perc']
df_ages_il.head()
sns.catplot(x='age', y='perc',hue='age', kind='bar', col='uf', col_wrap=2, data=df_ages_il, aspect=7, height=1.5)
plt.figure(figsize=(25, 10))

plt.grid = True

#sns.violinplot(y=df_ages_il.perc, x=df_ages_il.age, scale='count')

sns.set_style("whitegrid")

#sns.barplot(y=df_ages_il.perc, x=df_ages_il.age)

sns.lineplot(x='age',y='perc', hue='uf', data=df_ages_il, palette='bright')