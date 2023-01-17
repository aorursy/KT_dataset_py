import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
pyo.init_notebook_mode()
covid19 = pd.read_csv("../input/oficial-colombia-data-covid19-03032020-to-29082020/COVID_COL_31082020.csv") 
covid19.head()
covid19.tail()
covid19.shape
covid19.isnull().sum()
covid19.columns
covid19.describe()
covid19["Fecha de notificación"] = pd.to_datetime(covid19["Fecha de notificación"]) 
covid19["Fecha de notificación"].min()
covid19["Fecha de notificación"].max()
%matplotlib inline
k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.figure(figsize=(20,10)) 
plt.hist(covid19["Edad"], bins = k, align='left', color='b', edgecolor='red', linewidth=1)
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.title("# Casos Positivos COVID 19 COLOMBIA - Rangos de Edad")
covid_fallecidos = covid19[covid19['Fecha de muerte'].notna()]
# % Fallecidos % Dead People
print(len(covid_fallecidos)*100/len(covid19), " % de Fallecidos")
#Ajuste de Grafica - Graph Setup
k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.figure(figsize=(20,10)) 
plt.hist(covid_fallecidos["Edad"], bins = k, align='left', color='b', edgecolor='red', linewidth=1)
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.title("# Casos Fallecidos COVID 19 COLOMBIA - Rangos de Edad - 31/08/2020")
#Solo Recuperados confirmados, Excluye a: 'Fallecido', 'Hospital', 'Casa', nan, 'Hospital UCI', 'hospital'
covid_recuperados = covid19[covid19['atención']=='Recuperado']
print(len(covid_recuperados)*100/len(covid19)," % Recuperados Confirmados")
k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.figure(figsize=(20,10)) 
plt.hist(covid_recuperados["Edad"], bins = k, align='left', color='b', edgecolor='red', linewidth=1)
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.title("# Casos Recuperados COVID 19 COLOMBIA - Rangos de Edad - 31/08/2020")
#la variable genero se encuentra con diferentes valores se debe aplicar ajuste de datos
covid_recuperados["Sexo"].unique()
#Se Ajusta a mayuscula para unificar variable
covid_recuperados["Sexo"] = covid_recuperados["Sexo"].str.upper()
covid_recuperados["Sexo"].unique()
covid_recuperados['Sexo'].value_counts().plot(kind='bar')
diario_covid = covid19[['Fecha de notificación', ]].copy()
diario_covid["Count"]  = 1
diario_covid
diario_covid.rename(columns = {'Fecha de notificación':'Fecha'}, inplace = True)
diario_covid = diario_covid.groupby(pd.Grouper(key='Fecha',freq='D')).sum().reset_index()
diario_covid = diario_covid.sort_values('Fecha', ascending=True)
diario_covid = diario_covid.drop(diario_covid.index[[181]])
diario_covid
fig = go.Figure([go.Scatter(x=diario_covid['Fecha'], y=diario_covid['Count'])])
fig.show()
df_train = diario_covid.rename(columns={'Fecha': 'ds', 'Count': 'y'})
model = Prophet(interval_width = 0.95,
                yearly_seasonality = False,
                weekly_seasonality = False,
                daily_seasonality = True,
                holidays = None,
                changepoint_prior_scale = 0.05)
model.fit(df_train)
forecast = model.make_future_dataframe(periods=30, freq = 'D')
forecast = model.predict(forecast)
fig1 =model.plot(forecast)
fig2 = model.plot_components(forecast)
plt.show()
forecast.tail(30)
def_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
def_forecast = def_forecast.rename(columns={'ds': 'Fecha', 'yhat': 'Pronostico', 'yhat_lower': 'PronosticoMinimo', 'yhat_upper': 'PronosticoMaximo'})
def_forecast