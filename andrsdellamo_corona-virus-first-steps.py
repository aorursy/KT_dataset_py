# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install wget

import pandas as pd 
import numpy as np

#visualización
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

import wget
import os

from scipy.optimize import curve_fit
# Function that updates the data downloading it forn internet and formats it, Leaving it ready to work
# We call it like this:
# covid = get_coronavirus_data ()
def get_coronavirus_data ():
    urls=['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
          'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
          'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv']
    ficheros=["time_series_covid19_confirmed_global.csv",
              "time_series_covid19_deaths_global.csv",
              "time_series_covid19_recovered_global.csv"]
    for i in ficheros:
        if os.path.exists(i):
            os.remove(i)
    for url in urls:
        file_name=wget.download(url)
    cf_df=pd.read_csv('time_series_covid19_confirmed_global.csv')
    de_df=pd.read_csv('time_series_covid19_deaths_global.csv')
    re_df=pd.read_csv('time_series_covid19_recovered_global.csv')
    cf_df_trans=cf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                           value_vars=cf_df.columns[4:], 
                           var_name='Date', 
                           value_name='Confirmed')
    de_df_trans=de_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                           value_vars=de_df.columns[4:], 
                           var_name='Date', 
                           value_name='Deaths')
    re_df_trans=re_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                           value_vars=re_df.columns[4:], 
                           var_name='Date', 
                           value_name='Recovered')
    data_0=pd.merge(cf_df_trans,de_df_trans, how='inner', on=['Province/State', 'Country/Region', 'Lat', 'Long','Date'])
    data=pd.merge(data_0,re_df_trans, how='inner', on=['Province/State', 'Country/Region', 'Lat', 'Long','Date'])
    data["Date"]=data["Date"].apply(pd.to_datetime)
    data['Active']=data['Confirmed']-data['Deaths']-data['Recovered']
    data.rename(columns={'Country/Region':'Pais',
                         'Province/State':'Provincia',
                         'Date':'Fecha'},inplace=True)
    return data


# Function to generate columns with the daily increments of the Confirmed, Deaths ... fields
# It only does it for DataFrames filtered and only to one country
# We call you:
# deltac = generate_delta (covid_spain, 'Confirmed', 'DeltaC')
def generar_delta (datos,variable,nombre):
    j=0
    x0=0
    delta=[]
    for x in datos[variable]:
        if (j==0):
            x0=0
            x1=datos.loc[j,variable]
        else:
            x0=datos.loc[(j-1),variable]
            x1=datos.loc[(j),variable]
        delta=delta+[x1-x0]
        j+=1
    df=pd.DataFrame(delta,columns=[nombre])
    return df


# We add the column Day to be able to adjust with the function, since it does not admit dates, but numbers
# We add a sequential column for each country with the number of the day
# We call you:
# dias_df = genera_dia (covid_pais ['Pais']
def genera_dia (dato):
    j=0
    dias=[]
    for x in dato:
        if (j==0):
            pais=x
        if (pais==x):
            dias=dias+[j]
            pais=x
        else:
            j=0
            pais=x
            dias=dias+[j]
        j+=1
    df=pd.DataFrame(dias,columns=['Dia'])
    return df

# Exponential growth / decrease function with 3 parameters
# to adjust
def func(x, a, b, c):
    return  a*np.exp(b*(x - c))

# Function that does exponential regression for a list of given countries and then paints it
# its returns the adjust parameters.
def pintar_predicciones(datos,lista_paises,pintar,pintarbool,ndias):
    dparam={}
    dcov={}
    # we make the fit to the last function
    for x in lista_paises:
        popt, pcov = curve_fit(func, 
                           datos[(datos['Pais']==x)]['Dia'], 
                           datos[(datos['Pais']==x)][pintar], maxfev=30000)
        dparam[x]=popt
        dcov[x]=pcov
    # if pintargool is True, then we paint the fit
    if (pintarbool):
        dias = pd.DataFrame(np.arange(1,ndias)).rename(columns = {0:'dia'})
        fig = go.Figure()
        for x in lista_paises:
            # Paint the data
            fig.add_trace(go.Scatter(x=datos[(datos['Pais']==x)]['Dia'], 
                                 y=datos[(datos['Pais']==x)][pintar], 
                                 mode='lines+markers', name=pintar+' '+x))
            # Paint the fit
            fig.add_trace(go.Scatter(x=dias['dia'], 
                                 y=func(dias['dia'], *dparam[x]), 
                                 mode='lines', name='Prediccion '+x))
        fig.show()
    return dparam,dcov


covid=get_coronavirus_data()

covid_pais=covid.groupby(['Pais','Fecha'])[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()
covid_pais.sort_values(by=['Pais','Fecha'], ascending=['False','False'],inplace=True)

dias_df=genera_dia(covid_pais['Pais'])

dias_df.shape,covid_pais.shape

covid_pais=pd.concat((covid_pais,dias_df),axis=1)

lista_paises_pintar=['India','Brazil','Mexico','Chile','Peru','Iran','US']

mparan,mcov=pintar_predicciones(covid_pais,lista_paises_pintar,'Confirmed',True,220)
mparan,mcov=pintar_predicciones(covid_pais,lista_paises_pintar,'Deaths',True,220)
mparan,mcov=pintar_predicciones(covid_pais,lista_paises_pintar,'Active',True,220)
mparan,mcov=pintar_predicciones(covid_pais,lista_paises_pintar,'Confirmed',True,220)

mparan

parametros=pd.DataFrame(mparan)
parametros=parametros.T.reset_index()
parametros.drop(columns=[0,2],inplace=True)

parametros['TiempoTriplicar']=1/parametros[1]
parametros['TiempoDuplicar']=2/(np.e*parametros[1])
parametros.sort_values(by='TiempoDuplicar')
mparan,mcov=pintar_predicciones(covid_pais,
                                covid_pais.groupby(['Pais']).size().index.to_list(),
                                'Confirmed',False,220)
parametros=pd.DataFrame(mparan)
parametros=parametros.T.reset_index()
parametros.drop(columns=[0,2],inplace=True)
# Calculamos los tiempos de semicrecimiento (x3) y de duplicacion 
parametros['TiempoTriplicar']=1/parametros[1]
parametros['TiempoDuplicar']=2/(np.e*parametros[1])
parametros.sort_values(by='TiempoDuplicar')
mparan,mcov=pintar_predicciones(covid_pais,['Gambia'],'Confirmed',True,220)
mparan,mcov=pintar_predicciones(covid_pais,['Spain'],'Confirmed',True,220)