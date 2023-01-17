!pip install pycountry



import pandas as pd  #manipulación y análisis de datos 

import numpy as np   #soporte para vectores y matrices

import seaborn as sns #libreria para visualizacion

from matplotlib import pyplot as plt #biblioteca para generacion de graficos

import plotly.graph_objects as go #biblioteca para generacion de graficos interactivos

from fbprophet import Prophet #procedimientos para predicciones

import pycountry #provee funciones de conversion entre nombres de paises en formato ISO, codigos de pais y nombres de continentes. 

import plotly.express as px

from scipy.optimize import curve_fit #es una biblioteca libre y de código abierto para Python. Se compone de herramientas y algoritmos matemáticos. 

from datetime import date #funciones de tiempo y fechas
today = date.today()



# Mes, dia y año	

d2 = today.strftime("%d  %B, %Y")



print("Base de datos actualizada a la fecha ====> " + d2 + " <====")
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['Last Update'])

df.rename(columns = {'ObservationDate' : 'Date', 'Country/Region' : 'Country'}, inplace=True)



df_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")



df_confirmed.rename(columns = {'Country/Region' : 'Country'}, inplace=True)

df_recovered.rename(columns = {'Country/Region' : 'Country'}, inplace=True)

df_deaths.rename(columns = {'Country/Region' : 'Country'}, inplace=True)



df

#df_confirmed
countries = df['Country'].unique().tolist()

data_new = {}

for name in countries:

  a = df[df['Country'] == name].groupby('Date').sum()

  data_new[name] = a.to_numpy()

print("Total de Paises infectados con COVID19 a nivel mundial: ", len(countries))

from matplotlib.pylab import *



figure(figsize = (10, 7))

large_cases = []



countries_list = ["Mexico", "Colombia", "Argentina", "Peru", "Venezuela", "Chile", "Ecuador", "Guatemala", "Cuba", "Haiti", "Bolivia", "Dominican Republic", "Honduras", "Paraguay",

                  "Nicaragua", "El Salvador", "Costa Rica", "Panama", "Puerto Rico", "Uruguay", "Guadeloupe", "Martinique", "French Guiana", "Saint Martin", "Saint Barthélemy"]



countries_list = [ "Venezuela", "Ecuador", "Guatemala", "Cuba", "Haiti", "Bolivia", "Dominican Republic", "Honduras", "Paraguay",

                  "Nicaragua", "El Salvador", "Costa Rica", "Panama", "Puerto Rico", "Uruguay", "Guadeloupe", "Martinique",]



#Place names of countries in your scope and uncomment the following line

#countries_list = ["Bolivia", "Mexico",   "Colombia", "Costa Rica", "Venezuela", "Ecuador"]



count = 0



for name in countries: 

  confirm = data_new[name][:, 1]

  for country_name in countries_list:

    if name == country_name:

      plot(confirm, 'o-', label = name)



legend(fontsize = 10, loc = 'upper left')

xlabel("Número de días")

ylabel("Número de casos")

title('COVID19 Casos Confirmados en Paises de America Latina')

grid(True)

show()
def calc_covid_fit(name, days, start, stop):

  name_conf = data_new[name][:,1][start:stop]

  name_dea = data_new[name][:,2][start:stop]

  name_rec = data_new[name][:,3][start:stop]



  def function_to_fit(x, a, b, c, d):

    return a/(d + b * np.exp(-c * x))

  #print(name_conf)

  xdata = np.arange(0, len(name_conf), 1)

  x_100 = np.arange(0, stop + days, 1)

  popt, pcov = curve_fit(function_to_fit, xdata, name_conf)

  figure(figsize=(8,5))

  plot(x_100, function_to_fit(x_100, *popt), label = 'Model', linewidth = 3)

  plot(xdata, name_conf, 'o', label = "Confirmados")

  plot(xdata, name_dea, 'o', label = "Decesos", color = "Red")

  plot(xdata, name_rec, 'o', label = "Recuperados", color = "Green")

  legend()

  xlabel("Número de días")

  ylabel("Número de casos")

  title("COVID19 Casos Confirmados en " + name)

  grid(True)

  show()
calc_covid_fit("Mexico", 2, 0, 220)
calc_covid_fit("Bolivia", 2, 0, 220)
calc_covid_fit("Costa Rica", 2, 0,220)
calc_covid_fit("Peru", 2, 0,220)
cont = 1

latin_cases = []

latin_cases_values = []



#countries_list = ["Brazil", "Mexico", "Colombia", "Argentina", "Peru", "Venezuela", "Chile", "Ecuador", "Guatemala", "Cuba", "Haiti", "Bolivia", "Dominican Republic", "Honduras", "Paraguay",

#                  "Nicaragua", "El Salvador", "Costa Rica", "Panama", "Puerto Rico", "Uruguay", "Guadeloupe", "Martinique", "French Guiana", "Saint Martin", "Saint Barthelemy"]



countries_list = ["Mexico", "Colombia", "Argentina", "Peru", "Venezuela", "Chile", "Ecuador", "Guatemala", "Cuba", "Bolivia", "Paraguay",

                   "Uruguay", ]



#countries_list = ["Mexico", "Costa Rica","Colombia", "Argentina", "Peru", "Venezuela", "Chile", "Ecuador", "Bolivia", "Uruguay"]



for name in countries: 

  confirm = data_new[name][:, 1]

  for country_name in countries_list:

    if name == country_name:

      latin_cases.append(name)

      latin_cases_values.append(confirm[-1])



latin_cases_df = pd.DataFrame({ "Paises": latin_cases, "Casos": latin_cases_values})

latin_cases_df = latin_cases_df.sort_values(by = ["Casos"], ascending=False)





for names in range(len(latin_cases_df["Paises"].tolist())):

    latin_cases[names] = str(cont) + ". " + latin_cases_df.iloc[names]["Paises"] + " " + str(int(latin_cases_df.iloc[names]["Casos"]))

    cont += 1





fig, ax = plt.subplots( figsize=(15,8))



y_pos = np.arange(len(latin_cases_df["Casos"].tolist()))

ax.barh(y_pos, latin_cases_df["Casos"].tolist(), align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(latin_cases)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Numero de casos')

ax.set_title('COVID19 Casos Confirmados en Paises de America Latina')



plt.show()
