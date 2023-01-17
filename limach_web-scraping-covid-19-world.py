#Importando Pandas
import pandas as pd
#Upload da base de dados Covid 19 from GitHub
confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
# Salvando o dados em Dataframes
df_confirmed = pd.read_csv(confirmed)
df_death = pd.read_csv(death)
df_recovered = pd.read_csv(recovered)
# Verificando databeses
df_confirmed.shape
df_death.shape
df_recovered.shape
df_confirmed.head(5)
df_death.tail(5)
df_recovered.sample(5)
df_confirmed.columns
# Transpondo informações de data para linhas 
df_death = pd.melt(df_death, col_level=0, id_vars=["Country/Region","Province/State","Lat","Long"], var_name="Date",value_name="Death")
df_confirmed = pd.melt(df_confirmed, col_level=0, id_vars=["Country/Region","Province/State","Lat","Long"], var_name="Date",value_name="Confirmed")
df_recovered = pd.melt(df_recovered, col_level=0, id_vars=["Country/Region","Province/State","Lat","Long"], var_name="Date",value_name="Recovered")
df_confirmed.head()
df_recovered.head()
# Concatenando colunas de confirmados, mortos e recuperados em um unico Dataframe
df_world = pd.concat([df_confirmed,df_death,df_recovered], axis=1, ignore_index=False)
df_world.sample(10)
# Excluindo Colunas duplicadas
~df_world.columns.duplicated()
df_world = df_world.loc[:,~df_world.columns.duplicated()]
df_world.head()
# Corrigindo o formarto da date para 'datetime'
df_world['Date'] = pd.to_datetime(df_world.Date)
df_world['Date'].sample(10)
df_world['Date'].dt.strftime('%d/%m/%Y').sample(5)
#Database final
df_world.tail()
#Salvando database final em CSV
df_world.to_csv(r'E:\Python\Covid\Database\World\Covid_World.csv', index=0, sep='|')
#Analise e predição de Curva de Casos confirmados para para Brazil
import matplotlib.pyplot as plt

%matplotlib inline
#Importando Bibliotecas
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime, timedelta

import math
country = 'Brazil'

inicio = 0
# We want number of confirmed for each date for each country

country_data = df_world[df_world['Country/Region']==country]

#Dropando Colunas desnecessarias

country_data = country_data.drop(["Country/Region", "Province/State","Lat","Long","Recovered","Death"], axis=1)

country_data.tail()
country_data['Confirmed'].head()
country_graph = country_data['Confirmed'].reset_index()[inicio:]

country_graph.tail()
#Sigmoide Function 
# We will want x_data to be the number of days since first confirmed and the y_data to be the confirmed data.

# This will be the data we use to fit a logistic curve



y_data = country_graph['Confirmed']

x_data = np.arange(len(y_data))



def log_curve(x,ymax,x_0,k):

    return ymax / (1 + np.exp(-k*(x-x_0)))



# Fit the curve

popt, pocv = curve_fit(log_curve, x_data, y_data)

estimated_k, estimated_x_0, ymax = popt





# Plot the fitted curve

k = estimated_k

x_0 = estimated_x_0

y_fitted = log_curve(x_data, k, x_0, ymax)

print(k, x_0, ymax)





# Plot everything for illustration

fig = plt.figure(figsize=(20,4.5))

plt.plot(x_data, y_fitted, 'b-.', label='Estimativa')

plt.plot(x_data, y_data, 'ro', label='Casos Confirmados ')

plt.xlabel("Dias")

plt.ylabel("Total Infectados")

plt.legend(prop={'size': 10})

plt.title(country+"'s Data", size=15)

for (x,y) in zip(x_data,y_data):

    label = y

    plt.annotate(label, # this is the text

                (x,y), # this is the point to label

                textcoords="offset points", # how to position the text

                xytext=(0,5), # distance from text to points (x,y)

                ha='center') # horizontal alignment can be left, right or center

x_m = np.arange(len(y_data)+30)

y_m = log_curve(x_m, *popt)
#Previsão D+3 para o Brazil
previsao = pd.DataFrame([])
print(datetime.strftime(country_data["Date"].max(),"%d/%m/%Y"), country_data["Confirmed"].max())

for i in range(1,5):

    pday = (datetime.strftime(country_data["Date"].max()+timedelta(days=i),"%d/%m/%Y"))

    print(pday, math.ceil(y_m[len(y_data)+i]))

    previsao = previsao.append(pd.DataFrame({'Data': pday, 'Prev': math.ceil(y_m[len(y_data)+i])}, index=[0]), ignore_index=True)
previsao
#Salvando Previsão em CSV
previsao.to_csv(r'E:\Python\Covid\Database\World\Previsao_Brasil.csv', index=0, sep='|')
#Plotando Grafico Sigmoide 
fig = plt.figure(figsize=(20,5))

plt.plot(x_m, y_m, c='k', marker='x', label="Previsto")

plt.plot(x_data, y_data, c='r',marker="o", label = "Real")

plt.text(x_m[-1]+.5, y_m[-1], str(int(y_m[-1])), size = 10) #Máximo

plt.xlabel("Dias")

plt.ylabel("Total Infectados")

plt.legend(prop={'size': 10})

plt.title(country+"'s Data", size=15)

plt.savefig(r'E:\Python\Covid\Database\World\Covid_World.jpeg')

plt.show()


