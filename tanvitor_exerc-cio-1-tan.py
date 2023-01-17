import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
dfBR = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv')

dfBR.head()
dfBR.describe()
resposta = [['uf', 'qualitativa nominal'],['total_eleitores', 'quantitativa discreta'], ['f_16', 'quantitativa discreta'], ['f_17', 'quantitativa discreta'], 

            ['f_18_20', 'quantitativa discreta'], ['f_25_34', 'quantitativa discreta'], ['f_70_79', 'quantitativa discreta'], ['f_sup_79', 'quantitativa discreta'], 

            ['gen_feminino', 'qualitativa nominal'], ['gen_masculino', 'qualitativa nominal'], ['gen_nao_informado', 'qualitativa nominal']]



resposta = pd.DataFrame(resposta, columns = ['Variavel', 'Classificação'])



display(resposta)
dfBR['uf'].value_counts()
#OS GENEROS JÁ ESTAO SOMADOS 
!pip install plotly.express
import plotly.express as px
dfgroup = dfBR.groupby('uf',as_index=False)

dfsum=dfgroup.first()

dfsum=dfsum.sort_values(by='total_eleitores',ascending=False)
import plotly.express as px

fig = px.bar(dfsum, x='uf', y='total_eleitores')

fig.update_layout(title_text='TOTAL DE ELEITORES POR UF')

fig.show()
dfsum
import numpy as np

idade = np.array(['f_16', 'f_17', 'f_21_24', 'f_25_34','f_35_44','f_45_59','f_60_69','f_70_79', 'f_sup_79'])



total_idade = np.array([dfsum[i].sum() for i in idade])

total_idade
fig = px.bar(x=idade, y=total_idade)

fig.update_layout(title_text='Faixa de idade por eleitores')

fig.show()
masc=dfsum['gen_masculino'].sum()

fem= dfsum['gen_feminino'].sum()
import plotly.graph_objects as go

labels = ['gen_masculino','gen_feminino']

values = [masc, fem]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(title_text='COMPARAÇAO DA QUANTIDADE DE ELEITORES FEMININO E MASCULINO - DE TODAS AS UF ')

fig.show()
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)