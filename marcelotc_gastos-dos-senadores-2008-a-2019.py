import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('ggplot')

%matplotlib inline

%pylab inline
df = pd.read_csv("../input/dados-abertos-ceaps/CEAPS_2008_2019.csv")
df.info()
df.sample(10)
df.drop('DOCUMENTO', axis=1, inplace=True) 
df.isnull().any()
df.CNPJ_CPF = df.CNPJ_CPF.fillna('Nao Informado')

df.FORNECEDOR = df.FORNECEDOR.fillna('Nao Informado')

df.DETALHAMENTO = df.DETALHAMENTO.fillna('Nao Informado')

df.DATA = df.DATA.fillna('Nao Informado')
df[df['VALOR_REEMBOLSADO'].isnull()]
df.loc[[149869],'VALOR_REEMBOLSADO']=df.loc[[149869],'VALOR_REEMBOLSADO'].replace(NaN, 469.53)

df.loc[[149876],'VALOR_REEMBOLSADO']=df.loc[[149876],'VALOR_REEMBOLSADO'].replace(NaN, 460.45)
df.loc[[149869,149876]]
df.isnull().any()
df.info()
df[df['VALOR_REEMBOLSADO']<0]
df['VALOR_REEMBOLSADO'] = abs(df['VALOR_REEMBOLSADO'])
df['VALOR_REEMBOLSADO'].describe().round(2)
df['TIPO_DESPESA']=df['TIPO_DESPESA'].replace("Aluguel de imóveis para escritório político, compreendendo despesas concernentes a eles. ", "Aluguel imóveis para escritório político e despesas concernentes")
df['TIPO_DESPESA']=df['TIPO_DESPESA'].replace("Aquisição de material de consumo para uso no escritório político, inclusive aquisição ou locação de software, despesas postais, aquisição de publicações, locação de móveis e de equipamentos. " 

, "Material de consumo escritório, aquisição ou locação software, despesas postais, locação móveis e equiptos")
df['TIPO_DESPESA']=df['TIPO_DESPESA'].replace("Contratação de consultorias, assessorias, pesquisas, trabalhos técnicos e outros serviços de apoio ao exercício do mandato parlamentar", "Contratação consultorias, assessorias, pesquisas, trabalhos técnicos, outros serviços de apoio")
df['TIPO_DESPESA'].sample(10)
df['VALOR_REEMBOLSADO'].sum().round(2)
import cufflinks as cf

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import iplot



cf.go_offline()
ceaps_por_ano = df.groupby('ANO')['VALOR_REEMBOLSADO'].sum()



data = [go.Bar(x=ceaps_por_ano.index,

               y=ceaps_por_ano.values,

               marker = {'color': '#feca57',

                         'line': {'color': '#ff9f43',

                                  'width': 2}}, opacity= 0.5)]



# Layout

config_layout = go.Layout(title='Total de Gastos dos Senadores - CEAPS 2008 a 2019',

                                yaxis={'title':'VALOR R$(milhões)','range':[0, 30000000]},

                                xaxis={'title':''})

                                 

# Objeto figura

fig = go.Figure(data=data, layout=config_layout)



# plotando grafico

py.iplot(fig)
top10_2008 = df[df['ANO']==2008].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2009 = df[df['ANO']==2009].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2010 = df[df['ANO']==2010].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2011 = df[df['ANO']==2011].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2012 = df[df['ANO']==2012].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2013 = df[df['ANO']==2013].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2014 = df[df['ANO']==2014].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2015 = df[df['ANO']==2015].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2016 = df[df['ANO']==2016].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2017 = df[df['ANO']==2017].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2018 = df[df['ANO']==2018].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)

top10_2019 = df[df['ANO']==2019].groupby(['ANO','SENADOR'])['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10)
top10_total = pd.concat([top10_2008, top10_2009, top10_2010, top10_2011, top10_2012, top10_2012, top10_2013, top10_2014, top10_2015, 

                         top10_2016, top10_2017, top10_2018, top10_2019], sort=False)
df.groupby('SENADOR')['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).to_frame().head(10)
df.nlargest(10, 'VALOR_REEMBOLSADO')
df.groupby('SENADOR')['ANO'].value_counts()
top10_2019
top10_2019 = top10_2019.groupby('SENADOR').sum().sort_values(ascending=False)



data = [go.Bar(x=top10_2019.index,

               y=top10_2019.values,

               marker = {'color': 'lightblue',

                         'line': {'color': '#0abde3',

                                  'width': 2}}, opacity= 0.5)]



# Layout

config_layout = go.Layout(title='Os 10 Senadores que mais receberam reembolso CEAPS em 2019',

                                 yaxis={'title':'VALOR R$(MIL)','range':[0, 600000]},

                                 xaxis={'title':''})



# Objeto figura

fig = go.Figure(data=data, layout=config_layout)



# plotando grafico

py.iplot(fig)
df[df['ANO']==2019].nlargest(10, "VALOR_REEMBOLSADO")
df_mandato = df[['SENADOR','ANO']].drop_duplicates()

df_mandato = df_mandato.groupby('SENADOR').count()

df_mandato.rename(columns={'ANO': 'ANOS_MANDATO'}).sample(20) #Exemplo com 20 candidatos aleatórios
df.groupby(['SENADOR','ANO'])[['VALOR_REEMBOLSADO']].sum().head(20) # Exemplo com os 20 primeiros em ordem alfabética
top10_total.groupby(['SENADOR','ANO']).sum().sort_values(ascending=False)
df['TIPO_DESPESA'].value_counts()
df.groupby('TIPO_DESPESA')['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).round()
#2019

df_19 = df[df['ANO']==2019]

df_19.groupby('TIPO_DESPESA')['VALOR_REEMBOLSADO'].sum().sort_values(ascending=True).plot.barh()

plt.xlim(0,8000000)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.xlabel('Milhões', fontsize=14)

plt.title('Total de gastos por Categoria - 2019', backgroundcolor='#FFDD33')

plt.ylabel('')
sns.catplot(x="ANO", y="VALOR_REEMBOLSADO", kind='boxen', palette="Set2", data=df)

plt.title("Distribuição dos Valores Reembolsados 2008 a 2019")

plt.gcf().set_size_inches(12, 8)
from wordcloud import WordCloud

import matplotlib.pyplot as plt



text = open("../input/dados-abertos-ceaps/CEAPS_2008_2019.csv", encoding='utf-8').read()

wordcloud = WordCloud(max_font_size=100,width = 1520, height = 535).generate(text)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()