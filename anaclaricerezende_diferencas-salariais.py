# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py

import plotly.graph_objs as go

import seaborn as sns

import matplotlib.pyplot as plt



py.init_notebook_mode(connected=False)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/glassdoor-analyze-gender-pay-gap/Glassdoor Gender Pay Gap.csv')
df.head() #Dando uma olhadinha no nosso dataset
df.shape
df.describe() # Estatisticas descritivas
femaleDs = df[df['Gender'] == 'Female'].loc[:,['BasePay','JobTitle', 'Education', 'Seniority']]

maleDs = df[df['Gender'] == 'Male'].loc[:,['BasePay','JobTitle', 'Education', 'Seniority']]
print('Há {} pessoas do sexo feminino e {} do sexo masculino na base de dados'.format(len(femaleDs.index), len(maleDs.index)))

#Wohoo! Está até balanceado esse dataset para nossa analise 
#Sobre o desvio padrão

print('O desvio padrão da BasePay para o sexo feminino é de: {} e para masculino: {}'.format(np.std(femaleDs.BasePay), np.std(maleDs.BasePay)))
def DistPlot(x, color):

    fig, ax = plt.subplots()

    fig.set_size_inches(10, 6)

    sns.distplot(x, bins=100, color=color)
DistPlot(femaleDs.BasePay, 'blue')
DistPlot(maleDs.BasePay, 'green')
print('O maior salário para o sexo feminino é de: ${} e para masculino: ${}'.format(femaleDs.BasePay.max(), maleDs.BasePay.max()))
boxFemale = go.Box(y = femaleDs.BasePay,

                name = 'Feminino',

                marker = {'color': '#e74c3c'})

boxMale = go.Box(y = maleDs.BasePay,

                name = 'Masculino',

                marker = {'color': '#00a000'})



data = [boxFemale, boxMale]



layout = go.Layout(title = 'Dispersão de salário por gênero',

                   titlefont = {'family': 'Arial',

                                'size': 22,

                                'color': '#7f7f7f'},

                   xaxis = {'title': 'Gênero'},

                   yaxis = {'title': 'Salário'},

                   paper_bgcolor = 'rgb(243, 243, 243)',

                   plot_bgcolor = 'rgb(243, 243, 243)')





fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
sns.catplot(x="Gender", hue="Education", kind="count", data=df)
df.JobTitle.unique()
sns.catplot(x="Gender", hue="JobTitle", kind="count", data=df)
femaleDs = femaleDs.loc[(femaleDs['JobTitle'] == 'Graphic Designer') | (femaleDs['JobTitle'] == 'Data Scientist')]

maleDs = maleDs.loc[(maleDs['JobTitle'] == 'Graphic Designer') | (maleDs['JobTitle'] == 'Data Scientist')]
femaleDs.nlargest(15, "BasePay") 
maleDs.nlargest(15, "BasePay") 
boxFemale = go.Box(y = femaleDs.BasePay,

                name = 'Feminino',

                marker = {'color': '#e74c3c'})

boxMale = go.Box(y = maleDs.BasePay,

                name = 'Masculino',

                marker = {'color': '#00a000'})



data = [boxFemale, boxMale]



layout = go.Layout(title = 'Dispersão de salário por gênero',

                   titlefont = {'family': 'Arial',

                                'size': 22,

                                'color': '#7f7f7f'},

                   xaxis = {'title': 'Gênero'},

                   yaxis = {'title': 'Salário'},

                   paper_bgcolor = 'rgb(243, 243, 243)',

                   plot_bgcolor = 'rgb(243, 243, 243)')





fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
print("Por desencargo de consciência, temos {} pessoas do sexo feminino e {} do masculino".format(len(femaleDs.index),len(maleDs.index)))
sns.catplot(x="Seniority", y="BasePay", hue="JobTitle", data=femaleDs)

sns.catplot(x="Seniority", y="BasePay", hue="JobTitle", data=maleDs)