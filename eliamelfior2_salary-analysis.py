# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import csv
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

# plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

init_notebook_mode(connected=True) #do not miss this line
# Arquivo CSV que contém os salários de julho
file = '../input/July_Salaries.csv'
f = open(file,'rt')
reader = csv.reader(f)

# Coloca os salários numa lista
csv_list = []
for l in reader:
    csv_list.append(l)
f.close()

# Inicializa um dataframe com os salários da lista
df = pd.DataFrame(csv_list)

# Pega a primeira coluna e coloca como nome.
df.columns = df.iloc[0]
df = df.reindex(df.index.drop(0),)
df = df.drop(columns=[None])
df.head()

#Lista os valores únicos dos cargos.  
df.Cargo.unique()
#Conta a quantidade de cargos diferentes dos servidores #592
len(df.Cargo.unique())
#Lista os diferentes órgãos de exercício.  
df.OrgaoExercicio.unique()  
#Conta a quantidade de órgãos diferentes.  
len(df.OrgaoExercicio.unique())  
df.Cargo.value_counts().nlargest(10)
trace0 = go.Bar(
            x=['PROFESSOR ACT - COD HAB 300', 'PROFESSOR', 'PROFESSOR ACT - COD HAB 100', 'ANALISTA TECNICO EM GESTAO E PROMOCAO DE SAUDE', 'AGENTE PENITENCIARIO', '3 SARGENTO', 'ESTAGIARIO', 'CABO', 'SOLDADO 2 CLASSE', 'AGENTE DE POLICIA CIVIL'],
            y=[18703, 15840, 9384, 8969, 3154, 3061, 2858, 2616, 2606, 2228],
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
            opacity=0.6
    )

data = [trace0]
layout = go.Layout(
    title='10 profissões que mais aparecem',
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='text-hover-bar')

#Pega os salários dos professores ACT - COD HAB 300
ProfessorACT = df.loc[df['Cargo'] == 'PROFESSOR ACT - COD HAB 300']

# Converte uma coluna para numérico
SalariosProfessores = pd.to_numeric(ProfessorACT['ValorBruto'].str.replace(',','.'))

data = [go.Histogram(x=SalariosProfessores)]

py.offline.iplot(data, filename='basic histogram')
num_bins = 15
plt.hist(SalariosProfessores, num_bins, normed=1, facecolor='blue', alpha=0.5)
plt.show()
import seaborn as sns
from scipy.stats import norm

ax = sns.distplot(SalariosProfessores)

data = [go.Histogram(x=SalariosProfessores,
                     cumulative=dict(enabled=True))]

py.offline.iplot(data, filename='cumulative histogram')
SalariosProfessores.sum()

#Pega os salários dos professores ACT - COD HAB 300
ProfessorACT = df.loc[df['Cargo'] == 'PROFESSOR ACT - COD HAB 100']

# Converte uma coluna para numérico
SalariosProfessores2 = pd.to_numeric(ProfessorACT['ValorBruto'].str.replace(',','.'))

data2 = [go.Histogram(x=SalariosProfessores2)]

py.offline.iplot(data2, filename='basic histogram')
trace1 = go.Histogram(
    x=SalariosProfessores,
    name='Professores Código 300',
    opacity=0.75
)
trace2 = go.Histogram(
    x=SalariosProfessores2,
    name='Professores Código 100',
    opacity=0.75
)

data = [trace1, trace2]
layout = go.Layout(barmode='overlay')
fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig, filename='overlaid histogram')
df.ValorBruto = pd.to_numeric(df.ValorBruto.str.replace(',','.'))
ValorBrutoGrouped = df.groupby('Cargo').aggregate('mean').sort_values('ValorBruto', ascending=False)
ValorBrutoGrouped

# horizontal bar plot
ValorBrutoGrouped[0:9].plot.barh()

# show the plot
plt.show()
CargosEValores = df.groupby('Cargo').aggregate('sum').sort_values('ValorBruto', ascending=False)

labels = []
values = []

for index, row in CargosEValores.iterrows():
    labels.append(index)
    values.append(int(row))

labels = labels[0:9]
values = values[0:9]

tracePie = go.Pie(labels=labels, values=values)
py.offline.iplot([tracePie], filename='basic_pie_chart')
ÓrgãosEValores = df.groupby('OrgaoExercicio').aggregate('sum').sort_values('ValorBruto', ascending=False)

labels = []
values = []

for index, row in ÓrgãosEValores.iterrows():
    labels.append(index)
    values.append(int(row))

labels = labels[0:9]
values = values[0:9]

tracePie = go.Pie(labels=labels, values=values)
py.offline.iplot([tracePie], filename='basic_pie_chart')
df['ValorBruto'].sum()
df.nlargest(30, 'ValorBruto')
newDf = df.groupby('Cargo').aggregate('count')
newDf = newDf.drop(columns=['Nome', 'CPF', 'ValorBruto', 'OrgaoExercicio'])
newDf.columns = ['Quantidade']
newDf
newDf2 = df.groupby('Cargo').aggregate('sum')
newDf.columns = ['SomaValorBruto']
newDf2
grouper = df.groupby('Cargo')
res = grouper.count()
res['ValorBruto'] = grouper.sum()['ValorBruto']
res = res.drop(columns=['Nome', 'OrgaoExercicio', 'OrgaoOrigem'])
res

trace = go.Scatter(
    x = res['CPF'],
    y = res['ValorBruto'],
    mode = 'markers',
    text= res.index
)

data = [trace]

# Plot and embed in ipython notebook!
py.offline.iplot(data, filename='basic-scatter')
grouper = df.groupby('Cargo')
res = grouper.count()
res['ValorBruto'] = grouper.mean()['ValorBruto']
res = res.drop(columns=['Nome', 'OrgaoExercicio', 'OrgaoOrigem'])
res

trace = go.Scatter(
    x = res['CPF'],
    y = res['ValorBruto'],
    mode = 'markers',
    text= res.index
)

data = [trace]

# Plot and embed in ipython notebook!
py.offline.iplot(data, filename='basic-scatter')