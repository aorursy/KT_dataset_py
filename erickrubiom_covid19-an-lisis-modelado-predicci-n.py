#!pip install plotly
# Permite ajustar la anchura de la parte útil de la libreta (reduce los márgenes)
from IPython.core.display import display, HTML
display(HTML("<style>.container{ width:98% }</style>"))
#import warnings 
#warnings.filterwarnings("ignore")
#Cargar Datos con Pandas
import pandas as pd
from datetime import datetime
#from librerias.funciones import *

%matplotlib inline

filename = '/kaggle/input/covid19/data/time-series-19-covid-combined.csv'
data = pd.read_csv(filename)# , names=names)

data
#data.head(200)
#data.dtypes
#Agrupando datos
data=data[(data['Confirmed']>0)]
data=data.groupby(['Country/Region','Date'], as_index=False)['Confirmed','Recovered','Deaths'].sum()
data
import matplotlib.pyplot as plt
from datetime import datetime
def pintarGrafica(data,pais,dias,titulo):
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')
    
    filtered_data=data[(data['Country/Region']==pais) & (data['Confirmed']>0)]
    filtered_data=filtered_data.head(dias)
    rowId = [] 
    i=0
    for index, row in filtered_data.iterrows():
        i=i+1
        rowId.append(i)
    filtered_data['rowid']=rowId
    #x=rowId
    x = filtered_data['rowid'].values.reshape(-1, 1) 
    yConfirmed = filtered_data['Confirmed'].values.reshape(-1, 1)
    yRecovered = filtered_data['Recovered'].values.reshape(-1, 1)
    yDeaths = filtered_data['Deaths'].values.reshape(-1, 1)
    filtered_data
    plt.scatter(x, yConfirmed)
    plt.plot(x, yConfirmed,label=pais)
    plt.legend()
    
    plt.xlabel('Dias')
    plt.ylabel('Casos')
    plt.title(titulo)
    
    fechaPrimerCaso=datetime.strptime(filtered_data.Date.min(), '%Y-%m-%d')
    
    return x,yConfirmed,yRecovered,yDeaths,fechaPrimerCaso
    #plt.show()
    
def modeloPredictivo(data,pais,gc,gr,gd,fechaAnalisis):
    import numpy as np
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    
    filtered_data=data[(data['Country/Region']==pais) & (data['Confirmed']>0)]
    rowId = [] 
    i=0
    for index, row in filtered_data.iterrows():
        i=i+1
        rowId.append(i)
    filtered_data['rowid']=rowId    
    
    x = filtered_data['rowid'].values.reshape(-1, 1) 
    yConfirmed = filtered_data['Confirmed'].values.reshape(-1, 1)
    yRecovered = filtered_data['Recovered'].values.reshape(-1, 1)
    yDeaths = filtered_data['Deaths'].values.reshape(-1, 1)
    
    fechaPrimerCaso=datetime.strptime(filtered_data.Date.min(), '%Y-%m-%d')
    
    from sklearn.linear_model import LinearRegression

    modelConfirmed = LinearRegression()
    modelDeaths = LinearRegression()
    modelRecovered = LinearRegression()
    #Aplicamos fit y transform y graficamos los datos ajustados del entrenamiento.
    from sklearn.preprocessing import PolynomialFeatures

    polyConfirmed = PolynomialFeatures(degree=gc, include_bias=False) #aqui jugamos con el degree hasta que el R2 sea cercano a 1 y el rsme sea minimo.
    x_polyConfirmed = polyConfirmed.fit_transform(x.reshape(-1,1))
    modelConfirmed.fit(x_polyConfirmed, yConfirmed)
    yConfirmed_pred = modelConfirmed.predict(x_polyConfirmed)

    polyRecovered = PolynomialFeatures(degree=gr, include_bias=False) #aqui jugamos con el degree hasta que el R2 sea cercano a 1 y el rsme sea minimo.
    x_polyRecovered = polyRecovered.fit_transform(x.reshape(-1,1))
    modelRecovered.fit(x_polyRecovered, yRecovered)
    yRecovered_pred = modelRecovered.predict(x_polyRecovered)
    
    polyDeaths = PolynomialFeatures(degree=gd, include_bias=False) #aqui jugamos con el degree hasta que el R2 sea cercano a 1 y el rsme sea minimo.
    x_polyDeaths = polyDeaths.fit_transform(x.reshape(-1,1))
    modelDeaths.fit(x_polyDeaths, yDeaths)
    yDeaths_pred = modelDeaths.predict(x_polyDeaths)


    plt.scatter(x, yConfirmed,c='blue')
    plt.scatter(x, yRecovered,c='black')    
    plt.scatter(x, yDeaths,c='red')


    plt.plot(x, yConfirmed_pred, '-r',color='blue',label='Confirmed')
    plt.plot(x, yRecovered_pred, color='black',label='Recovered')    
    plt.plot(x, yDeaths_pred, color='red',label='Deaths')

    plt.xlabel('Dias')
    plt.ylabel('Casos')
    plt.title('Corona Virus')

    plt.legend()

    plt.show()

    rmseConfirmed = np.sqrt(mean_squared_error(yConfirmed,yConfirmed_pred))
    r2Confirmed = r2_score(yConfirmed,yConfirmed_pred)
    
    rmseRecovered = np.sqrt(mean_squared_error(yRecovered,yRecovered_pred))
    r2Recovered = r2_score(yRecovered,yRecovered_pred)
    
    rmseDeaths = np.sqrt(mean_squared_error(yDeaths,yDeaths_pred))
    r2Deaths = r2_score(yDeaths,yDeaths_pred)
    
    #Calculamos el error del modelo
    print('rmseConfirmed: ' + str(rmseConfirmed))
    print('r2Confirmed: ' + str(r2Confirmed))
    print('------')
    print('rmseRecovered: ' + str(rmseRecovered))
    print('r2Recovered: ' + str(r2Recovered))    
    print('------')
    print('rmseDeaths: ' + str(rmseDeaths))
    print('r2Deaths: ' + str(r2Deaths))
    print('------')
    
    #fechaPrimerCaso=datetime.strptime(filtered_data.Date.min(), '%Y-%m-%d')
    #from datetime import datetime
    fechaAnalisisCaso=datetime.strptime(fechaAnalisis, '%Y-%m-%d')
    dias=(fechaAnalisisCaso-fechaPrimerCaso).days+1
    #print(dias)
    #print('-----------------------------')
    #print(fechaAnalisisCaso)
    #print('-----------------------------')
    X_test=np.array([[dias]])
    y_test_Confirmed = modelConfirmed.predict(polyConfirmed.fit_transform(X_test))
    #print('Confirmed:', y_test_Confirmed)

    y_test_Recovered = modelRecovered.predict(polyRecovered.fit_transform(X_test))
    #print('Recovered:',y_test_Recovered)    
    
    y_test_Deaths = modelDeaths.predict(polyDeaths.fit_transform(X_test))
    #print('Deaths:',y_test_Deaths)


    #print('')
    #print('')
    #return y_test_Confirmed,y_test_Deaths,y_test_Recovered
    
    return y_test_Confirmed,y_test_Recovered,y_test_Deaths,filtered_data,fechaAnalisis
dias=50
#pintarGrafica(data,'China',dias)
titulo="COVID-19 Comparativo Mundial (Primeros Días de Contagio)"
pintarGrafica(data,'Peru',dias,titulo)
pintarGrafica(data,'Italy',dias,titulo)
pintarGrafica(data,'Spain',dias,titulo)
pintarGrafica(data,'France',dias,titulo)
#pintarGrafica(data,'US',dias,titulo)
plt.show()
dias=60
titulo="COVID-19 Comparativo Sudamericano (Primeros Días de Contagio)"
pintarGrafica(data,'Chile',dias,titulo)
pintarGrafica(data,'Brazil',dias,titulo)
pintarGrafica(data,'Ecuador',dias,titulo)
pintarGrafica(data,'Peru',dias,titulo)
pintarGrafica(data,'Colombia',dias,titulo)
pintarGrafica(data,'Argentina',dias,titulo)
pintarGrafica(data,'Uruguay',dias,titulo)
plt.show()
y_test_Confirmed,y_test_Recovered,y_test_Deaths,filtered_data,fechaAnalisis=modeloPredictivo(data,'Peru',5,3,3,'2020-04-11')

print('FECHA ANALISIS: ' + str(fechaAnalisis))
print('============================')
print('')
print('CONFIRMADOS:' + str(y_test_Confirmed))
print('RECUPERADOS:' + str(y_test_Recovered))
print('FALLECIDOS:' + str(y_test_Deaths))
data.head(100)
import plotly.express as px
data = data[data['Confirmed']>0]
data = data.groupby(['Date','Country/Region']).sum().reset_index()
fig = px.choropleth(data, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Confirmed", 
                    color_continuous_scale="blues",
                    hover_name="Country/Region", 
                    animation_frame="Date"
                   )
fig.update_layout(
    title_text = '<b>Propagación Mundial del Coronavirus<br>(Casos Confirmados)</b>',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()
data = data[data['Confirmed']>0]
data = data.groupby(['Date','Country/Region']).sum().reset_index()
fig = px.choropleth(data, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Recovered", 
                    color_continuous_scale="greens",
                    hover_name="Country/Region", 
                    animation_frame="Date"
                   )
fig.update_layout(
    title_text = '<b>Propagación Mundial del Coronavirus<br>(Casos Recuperados)</b>',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()
data = data[data['Confirmed']>0]
data = data.groupby(['Date','Country/Region']).sum().reset_index()
fig = px.choropleth(data, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Deaths", 
                    color_continuous_scale="reds",
                    hover_name="Country/Region", 
                    animation_frame="Date"
                   )
fig.update_layout(
    title_text = '<b>Propagación Mundial del Coronavirus<br>(Casos Fallecidos)</b>',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()