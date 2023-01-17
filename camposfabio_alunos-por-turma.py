# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from uteis_educacao import le_arquivos

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
horas = le_arquivos( '/kaggle/input/dados-educacao-turma/horas_aula/', 'H')

turma = le_arquivos( '/kaggle/input/dados-educacao-turma/alunos_turma/', 'A')

horas.keys()
#turma.keys()

#t = turma['2007']

#t[(t['Regiao'] == 'Norte') & ]
medias_A = pd.DataFrame()

medias_H = pd.DataFrame()



group_A = {}

group_H = {}



#Totais

for ano in turma.keys():   

    medias_A[ano] = turma[ano].mean()

    group_A[ano] = turma[ano].groupby(['Regiao']).mean()



#Totais

for ano in horas.keys():   

    medias_H[ano] = horas[ano].mean()    

    group_H[ano] = horas[ano].groupby(['Regiao']).mean() 
group_H

#medias = medias.transpose()

#medias.info()

#totais = medias[['Inf_Total','Fund_Total','Medio_Total']]

#medias.loc[['Inf_Total','Fund_Total','Medio_Total']]
totais_A = medias_A.loc[['Inf_Total','Fund_Total','Medio_Total']]

totais_H = medias_H.loc[['Inf_Total','Fund_Total','Medio_Total']]



totais_A.rename(index={'Inf_Total':'Infantil', 'Fund_Total':'Fundamental', 'Medio_Total':'Medio'},inplace=True)

totais_H.rename(index={'Inf_Total':'Infantil', 'Fund_Total':'Fundamental', 'Medio_Total':'Medio'},inplace=True)

totais_A = totais_A.transpose()

totais_H = totais_H.transpose()



totais_A.to_csv('totais_alunos_turma.csv')

totais_H.to_csv('totais_horas_aula.csv')
import matplotlib.pyplot as plt

import seaborn as sns



sns.set()

plt.tight_layout()

plt.rcParams['figure.figsize'] = (15,5)



totais_A.plot.bar(title='Média de Alunos por Turma',stacked=True).grid(axis='y')

totais_H.plot.bar(title='Média de Horas/Aula',stacked=True).grid(axis='y')

fig, ax = plt.subplots(3,2)

#plt.rcParams['figure.figsize'] = (5,10)

plt.tight_layout()

#fig.suptitle('Média de Alunos por Turma', fontsize=16)



plt.subplot(3,2,1)

totais_A['Infantil'].plot.line(color='green', title='Média de Alunos por Turma').grid(axis='y')

plt.legend(['Infantil'], loc='center right')



plt.subplot(3,2,2)

totais_H['Infantil'].plot.line(color='green', title='Média de Horas/Aula').grid(axis='y')

plt.legend(['Infantil'], loc='center right')



plt.subplot(3,2,3)

totais_A['Fundamental'].plot.line(color='orange').grid(axis='y')

plt.legend(['Fundamental'], loc='center right')



plt.subplot(3,2,4)

totais_H['Fundamental'].plot.line(color='orange').grid(axis='y')

plt.legend(['Fundamental'], loc='center right')



plt.subplot(3,2,5)

totais_A['Medio'].plot.line(color='blue').grid(axis='y')

plt.legend(['Médio'], loc='center right')



plt.subplot(3,2,6)

totais_H['Medio'].plot.line(color='blue').grid(axis='y')

plt.legend(['Medio'], loc='center right')
from bokeh.plotting import figure, output_file, show, ColumnDataSource, output_notebook

from bokeh.models import HoverTool

from bokeh.layouts import column



output_notebook()

alunos = figure(x_axis_label='Ano', y_axis_label='Alunos', title='Alunos por Sala')



source_A = ColumnDataSource(totais_A)



alunos.line(x = 'index', y = 'Infantil', source=source_A, color='red')

alunos.line(x = 'index', y = 'Fundamental', source=source_A, color='blue')

alunos.line(x = 'index', y = 'Medio', source=source_A, color='green')



aInf = alunos.circle(x = 'index', y = 'Infantil', source=source_A, color='red', fill_color='white', size=4, legend='Infantil ')

aFund = alunos.circle(x = 'index', y = 'Fundamental', source=source_A, color='blue', fill_color='white',size=4, legend='Fundamental ')

aMed = alunos.circle(x = 'index', y = 'Medio', source=source_A, color='green', fill_color='white',size=4, legend='Médio ')



alunos.add_tools(HoverTool(renderers=[aInf],  tooltips=[("Ano", "@index"), ('Alunos', '@Infantil')]))

alunos.add_tools(HoverTool(renderers=[aFund], tooltips=[("Ano", "@index"),('Alunos', '@Fundamental')]))

alunos.add_tools(HoverTool(renderers=[aMed],  tooltips=[("Ano", "@index"),('Alunos', '@Medio')]))





horas = figure(x_axis_label='Ano', y_axis_label='Horas', title='Horas/Aula')

source_H = ColumnDataSource(totais_H)



horas.line(x = 'index', y = 'Infantil', source=source_H, color='red')

horas.line(x = 'index', y = 'Fundamental', source=source_H, color='blue')

horas.line(x = 'index', y = 'Medio', source=source_H, color='green')



hInf = horas.circle(x = 'index', y = 'Infantil', source=source_H, color='red', fill_color='white', size=4, legend='Infantil ')

hFund = horas.circle(x = 'index', y = 'Fundamental', source=source_H, color='blue', fill_color='white',size=4, legend='Fundamental ')

hMed = horas.circle(x = 'index', y = 'Medio', source=source_H, color='green', fill_color='white',size=4, legend='Médio ')



horas.add_tools(HoverTool(renderers=[hInf],  tooltips=[("Ano", "@index"), ('Horas', '@Infantil')]))

horas.add_tools(HoverTool(renderers=[hFund], tooltips=[("Ano", "@index"),('Horas', '@Fundamental')]))

horas.add_tools(HoverTool(renderers=[hMed],  tooltips=[("Ano", "@index"),('Horas', '@Medio')]))





show(column(alunos, horas))
