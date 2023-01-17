import pandas as pd

df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv', delimiter=',')



resposta1 = [

    ["cod_municipio_tse", "Quantitativa Discreta"],["nome_municipio","Qualitativa Nominal"],

    ["uf","Qualitativa Nominal"],["total_eleitores","Quantitativa Discreta"],

    ["gen_feminino","Quantitativa Discreta"],["gen_masculino","Quantitativa Discreta"],

    ["gen_nao_informado","Quantitativa Discreta"]]

resposta1 = pd.DataFrame(resposta1, columns=["Variavel", "Classificação"])

resposta1
uf = pd.DataFrame(pd.value_counts(df["uf"]))

uf
n_mun = pd.DataFrame(pd.value_counts(df["nome_municipio"]))

n_mun
import matplotlib.pyplot as plt

import seaborn as sns

#Distribuição dos códigos de cada municipio informado no dataset

sns.distplot(df['cod_municipio_tse'])
title = "Quantidade de registros por UF"

plt.figure(figsize=(16,8))

df["uf"].value_counts().plot(kind = 'barh', title = title)
#Análise\contagem realizada com o principal estado de cada região do Brasil 

a = df.loc[(df['uf'] == 'SP') | (df['uf'] == 'PR') | (df['uf'] == 'MT') | (df['uf'] == 'MA') | (df['uf'] == 'AM')]



sns.countplot(a['uf'])

titulo = "Cidades que mais aparecem no dataset(UFs diferentes)"

df["nome_municipio"].value_counts().head(10).plot(kind = 'bar', title = titulo)
title = "Total de eleitores(Milhões)  x UF"

df.groupby('uf')['total_eleitores'].sum().reset_index().plot(kind = 'bar', x='uf', y='total_eleitores', title = title, figsize=(13,4))
title = "Eleitoras(Milhões)  x UF"

df.groupby('uf')['gen_feminino'].sum().reset_index().plot(kind = 'bar', x='uf', y='gen_feminino', title = title, figsize=(13,4))
sns.catplot(

    data=df,

    x='uf',

    y='gen_masculino',

    kind='box',

    height=5, 

    aspect=3,

    color='blue')



#Podemos observar grandes outliers em SP e RJ, principalmente.
title = "Total de votos com Gênero não informado(Milhões)  x UF"

df.groupby('uf')['gen_nao_informado'].sum().reset_index().plot(kind = 'bar', x='uf', y='gen_nao_informado', title = title, figsize=(13,4))

#Retirar as UFs com zero
df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv', delimiter=',')

df.head(10)