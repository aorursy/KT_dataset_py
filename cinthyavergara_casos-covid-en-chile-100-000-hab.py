# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# This is a COVID19 visualization considering the number of cases in Chile per 100,000 inhabitants 
# versus others countries like Germany, United Kingdom , US, France and Brazil. 

# Data is obtained from the Johns Hopkins University Center for Systems Science and Engineering (CSSE)
# repository in Github
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
%matplotlib inline 
df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv', parse_dates=['Date'])
paises = ['Chile', 'Germany', 'United Kingdom', 'US', 'France', 'Brazil']
df = df[df['Country'].isin(paises)]
#Crear columna de resumen
df['Cases'] = df[['Confirmed', 'Recovered', 'Deaths']].sum(axis=1)
# Estructurar datos
df = df.pivot(index='Date', columns='Country', values='Cases')
paises = list(df.columns)
covid = df.reset_index('Date')
covid.set_index(['Date'], inplace=True)
covid.columns = paises
# Calcular tasa por 100,000 habitantes - obtenidos de worldometers año 2020
habitantes = {'Chile':19117918 , 'Germany': 83786725  , 'United Kingdom': 67889726 , 'US': 330548815, 'France': 65275010, 'Brazil':212575200}
percapita = covid.copy()
for country in list(percapita.columns): 
  percapita[country] = percapita[country]/habitantes[country]*100000
# Colores y estilos por país 
colores = {'Chile':'#045275', 'Brazil':'#089099', 'France':'#7CCBA2', 'Germany':'#FCDE9C', 'US':'#DC3977', 'United Kingdom':'#7C1D6F'}
plt.style.use('fivethirtyeight')
percapitaplot = percapita.plot(figsize=(15,8), color=list(colores.values()), linewidth=2, legend=False)
percapitaplot.grid(color='#d4d4d4')
percapitaplot.set_xlabel('Month')
percapitaplot.set_ylabel('N° of Cases per 100,000 inhabitants')
for country in list(colores.keys()): 
  percapitaplot.text(x = percapita.index[-1], y = percapita[country].max(), color = colores[country], s = country, weight = 'bold')
percapitaplot.set_title("Cases Per Capita COVID-19\nUSA, Brasil, Alemania, Francia, UK  Chile\n CONSIDERING Current Cases, Recoveries y Deaths")
percapitaplot.text(x = percapita.index[1], y = -100,s = 'datagy.io Source: https://github.com/datasets/covid-19/blob/master/data/countries-aggregated.csv', fontsize = 10)