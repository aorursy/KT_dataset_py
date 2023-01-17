import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotagem

import sklearn as sk # Machine learning

import seaborn as sbn # gráficos

import ipywidgets as widgets

from IPython.display import display

#sudeste = pd.read_csv("../input/hourly-weather-surface-brazil-southeast-region/sudeste.csv")

#sudeste.head()
#sudeste.shape
#fields = ['city','prov','date','yr','mo', 'da', 'hr', 'temp']

#sudeste_r = sudeste[fields]

#sudeste_r.head(10)
#sudeste_r.shape
#sudeste_recent = sudeste_r[(sudeste_r['yr'] > 2013)]

#sudeste_recent.head()
#sudeste_recent.to_csv('sudeste_recent.csv',index=False)
sudeste_recent = pd.read_csv("../input/dataset-sudeste-recent/sudeste_recent.csv")
sudeste_recent.shape
sudeste_recent_temp = sudeste_recent['temp']

sudeste_recent_temp.replace(0, np.nan, inplace=True)

sudeste_recent_temp.interpolate('linear', inplace=True, limit_direction='both')
sudeste_recent['temp'] = sudeste_recent_temp
sudeste_recent.head()
sudeste_agrup_geral = sudeste_recent.groupby(['mo','da', 'hr']).mean().reset_index()

sudeste_agrup_geral.head()
sudeste_agrup_geral.shape
sudeste_agrup_prov = sudeste_recent.groupby(['prov','mo','da', 'hr']).mean().reset_index()

sudeste_agrup_prov.head()
sudeste_agrup_prov.shape
sudeste_agrup_prov['prov'].value_counts()
Lista_Estados=list(sudeste_agrup_prov['prov'].value_counts().index)

Lista_Estados
sudeste_agrup_city = sudeste_recent.groupby(['city','prov','mo','da', 'hr']).mean().reset_index()

sudeste_agrup_city.head()
sudeste_agrup_city.shape
sudeste_agrup_city['city'].value_counts()
Lista_Cidades=sorted(list(sudeste_agrup_city['city'].value_counts().index))

Lista_Cidades
def monthToNum(Month):

    return {

            'janeiro' : 1,

            'fevereiro' : 2,

            'marco' : 3,

            'abril' : 4,

            'maio' : 5,

            'junho' : 6,

            'julho' : 7,

            'agosto' : 8,

            'setembro' : 9, 

            'outubro' : 10,

            'novembro' : 11,

            'dezembro' : 12

    }[Month]
cidade_drop = widgets.Dropdown(

    options=Lista_Cidades,

    value='Afonso Cláudio',

    description='Cidade:',

)

display(cidade_drop)
#mes = 'janeiro'

def gera_plot(mes):

    temp_month_aux = sudeste_agrup_geral['temp']

    temp_month = temp_month_aux[(monthToNum(mes)-1)*720:monthToNum(mes)*720]

    plt.rcParams['figure.figsize'] = (14, 6)

    tick = list(range(31))

    ticks = [(n*24+(monthToNum(mes)-1)*720) for n in tick]

    fig = plt.figure()

    ax = fig.add_axes([0, 0, 1, 1]) # main axes

    ax.plot(temp_month)

    ax.set_xlabel('dia',fontsize=20)

    ax.set_ylabel('Temperatura °C',fontsize=20)

    ax.set_title('Temperatura do mês: '+mes, fontsize=28)

    ax.set_xticks(ticks)

    tick = [n+1 for n in tick]

    ax.set_xticklabels(map(str,tick))

    plt.show()
mes_drop = widgets.Dropdown(

    options=['janeiro','fevereiro', 'marco', 'abril', 'maio', 'junho', 

             'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'],

    value='janeiro',

    description='Mes:',

)

widgets.interact(gera_plot,mes = mes_drop)
import math



mes = mes_drop.value

temp_month_aux = sudeste_agrup_geral['temp']

temp_month = temp_month_aux[(monthToNum(mes)-1)*720:monthToNum(mes)*720]





ZT_bite = 0.9        #Figura de mérito do material BiTe

K = 273.15

Th = 35.5            #Temperatura do Corpo humano [°C]

Th = Th + K          #Temperatura do Corpo humano [K]

nTEGs = 12           #Número de TEGs em //

Rmteg = 3.8/nTEGs    #Resistência do conjunto de TEGs [ohm]

Smteg = 0.095        #Coeficiente de Seeback do TEG [V.K^-1]

Pap = 0.00171        #Potência Cosumida pelo aparelho [W]

CargaMax = 51.84     #Carga Máxima da Bateria/Cap [J]  (Bateria rec: 51.84J) (Bateria Cap: 17.1072J)

Eff_bateria = 0.75    #Eficiência de Carregamento da bateria

Preal = []           #List para guardar valores de potência gerada pelo TEG

Pef = []

Energia = [CargaMax] #List para guardar valores de energia  



for n in range(len(temp_month)):

    Tl = temp_month[n+720*(monthToNum(mes)-1)]+K

    deltaT = Th - Tl

    eta_ideal = 1 - math.sqrt(Tl/Th)

    eta_real = eta_ideal*ZT_bite/((1+math.sqrt(1+ZT_bite))**2)

    Voc = Smteg*deltaT

    Preal.append(eta_real*Voc**2/(4*Rmteg))

    Pef.append(Preal[n]-Pap)

    if n%24 >= 7 or n%24 <= 21:        #Horario pessoa acordada

        Energia.append(Energia[n]+(Preal[n]*Eff_bateria-Pap)*60*60)

    elif n%24 < 7 or n%24 > 21:        #Horario pessoa dormindo

        #Energia.append(Energia[n]+(Preal[n])*60*60)  #carrega enquanto dorme

        Energia.append(Energia[n])                    #não carrega enquanto dorme

    if Energia[n+1] > (CargaMax):

        Energia[n+1] = CargaMax

    elif Energia[n+1] < 0:

        Energia[n+1] = 0

    
tick = list(range(31))

ticks = [(n*24) for n in tick]

fig = plt.figure()

ax = fig.add_axes([0, 0, 0.8, 0.8]) # main axes

ax.plot(Energia)

ax.set_xlabel('dia',fontsize=20)

ax.set_ylabel('Carga da Bateria [J]',fontsize=20)

ax.set_title('Carga da Bateria no mês: '+mes, fontsize=28)

ax.set_xticks(ticks)

tick = [n+1 for n in tick]

ax.set_xticklabels(map(str,tick))

plt.show()
tick = list(range(31))

ticks = [(n*24) for n in tick]

fig = plt.figure()

ax = fig.add_axes([0, 0, 0.8, 0.8]) # main axes

Pef_mW = [n*100 for n in Pef]

ax.plot(Pef_mW)

ax.set_xlabel('dia',fontsize=20)

ax.set_ylabel('Potência Eficaz [mW]',fontsize=20)

ax.set_title('Potência Eficaz gerada ao longo do mês: '+mes, fontsize=28)

ax.set_xticks(ticks)

tick = [n+1 for n in tick]

ax.set_xticklabels(map(str,tick))

plt.show()
def gera_plot_prov(mes,estado):

    sudeste_prov = sudeste_agrup_prov[(sudeste_agrup_prov['prov'] == estado)]

    temp_month_aux = sudeste_prov['temp']

    temp_month_aux.reset_index(drop=True, inplace = True)

    temp_month = temp_month_aux[(monthToNum(mes)-1)*720:monthToNum(mes)*720]

    plt.rcParams['figure.figsize'] = (14, 6)

    tick = list(range(31))

    ticks = [(n*24+(monthToNum(mes)-1)*720) for n in tick]

    fig = plt.figure()

    ax = fig.add_axes([0, 0, 1, 1]) # main axes

    ax.plot(temp_month)

    ax.set_xlabel('dia',fontsize=20)

    ax.set_ylabel('Temperatura °C',fontsize=20)

    ax.set_title('Temperatura do estado '+ estado + ' no mês: '+mes, fontsize=28)

    ax.set_xticks(ticks)

    tick = [n+1 for n in tick]

    ax.set_xticklabels(map(str,tick))

    plt.show()
mes_drop = widgets.Dropdown(

    options=['janeiro','fevereiro', 'marco', 'abril', 'maio', 'junho', 

             'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'],

    value='janeiro',

    description='Mes:',

)

estado_drop = widgets.Dropdown(

    options=Lista_Estados,

    value='ES',

    description='Estado:',

)

widgets.interact(gera_plot_prov,mes = mes_drop,estado = estado_drop)
import math

ZT_bite = 0.9        #Figura de mérito do material BiTe

K = 273.15

Th = 35.5            #Temperatura do Corpo humano [°C]

Th = Th + K          #Temperatura do Corpo humano [K]

nTEGs = 12           #Número de TEGs em //

Rmteg = 3.8/nTEGs    #Resistência do conjunto de TEGs [ohm]

Smteg = 0.095        #Coeficiente de Seeback do TEG [V.K^-1]

Pap = 0.00171        #Potência Cosumida pelo aparelho [W]

CargaMax = 51.84     #Carga Máxima da Bateria/Cap [J]  (Bateria rec: 51.84J) (Bateria Cap: 17.1072J)

Eff_bateria = 0.75    #Eficiência de Carregamento da bateria

Preal = []           #List para guardar valores de potência gerada pelo TEG

Pef = []

Energia = [CargaMax] #List para guardar valores de energia  



for n in range(len(temp_month)):

    Tl = temp_month[n+720*(monthToNum(mes)-1)]+K

    deltaT = Th - Tl

    eta_ideal = 1 - math.sqrt(Tl/Th)

    eta_real = eta_ideal*ZT_bite/((1+math.sqrt(1+ZT_bite))**2)

    Voc = Smteg*deltaT

    Preal.append(eta_real*Voc**2/(4*Rmteg))

    Pef.append(Preal[n]-Pap)

    if n%24 >= 7 or n%24 <= 21:        #Horario pessoa acordada

        Energia.append(Energia[n]+(Preal[n]*Eff_bateria-Pap)*60*60)

    elif n%24 < 7 or n%24 > 21:        #Horario pessoa dormindo

        #Energia.append(Energia[n]+(Preal[n])*60*60)  #carrega enquanto dorme

        Energia.append(Energia[n])                    #não carrega enquanto dorme

    if Energia[n+1] > (CargaMax):

        Energia[n+1] = CargaMax

    elif Energia[n+1] < 0:

        Energia[n+1] = 0

    
tick = list(range(31))

ticks = [(n*24) for n in tick]

fig = plt.figure()

ax = fig.add_axes([0, 0, 0.8, 0.8]) # main axes

ax.plot(Energia)

ax.set_xlabel('dia',fontsize=20)

ax.set_ylabel('Carga da Bateria [J]',fontsize=20)

ax.set_title('Carga da Bateria do estado ' +estado_drop.value+ ' no mês: '+mes_drop.value, fontsize=28)

ax.set_xticks(ticks)

tick = [n+1 for n in tick]

ax.set_xticklabels(map(str,tick))

plt.show()
tick = list(range(31))

ticks = [(n*24) for n in tick]

fig = plt.figure()

ax = fig.add_axes([0, 0, 0.8, 0.8]) # main axes

Pef_mW = [n*100 for n in Pef]

ax.plot(Pef_mW)

ax.set_xlabel('dia',fontsize=20)

ax.set_ylabel('Potência Eficaz [mW]',fontsize=20)

ax.set_title('Potência Eficaz no estado ' +estado_drop.value+ ' gerada ao longo do mês: '+mes_drop.value, fontsize=28)

ax.set_xticks(ticks)

tick = [n+1 for n in tick]

ax.set_xticklabels(map(str,tick))

plt.show()
def gera_plot_city(mes,cidade):

    sudeste_city = sudeste_agrup_city[(sudeste_agrup_city['city'] == cidade)]

    temp_month_aux = sudeste_city['temp']

    temp_month_aux.reset_index(drop=True, inplace = True)

    temp_month = temp_month_aux[(monthToNum(mes)-1)*720:monthToNum(mes)*720]

    plt.rcParams['figure.figsize'] = (14, 6)

    tick = list(range(31))

    ticks = [(n*24+(monthToNum(mes)-1)*720) for n in tick]

    fig = plt.figure()

    ax = fig.add_axes([0, 0, 1, 1]) # main axes

    ax.plot(temp_month)

    ax.set_xlabel('dia',fontsize=20)

    ax.set_ylabel('Temperatura °C',fontsize=20)

    ax.set_title('Temperatura da cidade '+ cidade + ' no mês: '+mes, fontsize=28)

    ax.set_xticks(ticks)

    tick = [n+1 for n in tick]

    ax.set_xticklabels(map(str,tick))

    plt.show()
mes_drop = widgets.Dropdown(

    options=['janeiro','fevereiro', 'marco', 'abril', 'maio', 'junho', 

             'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'],

    value='janeiro',

    description='Mes:',

)

cidade_drop = widgets.Dropdown(

    options=Lista_Cidades,

    value='Afonso Cláudio',

    description='Cidade:',

)

widgets.interact(gera_plot_city,mes = mes_drop,cidade = cidade_drop)
import math

ZT_bite = 0.9        #Figura de mérito do material BiTe

K = 273.15

Th = 35.5            #Temperatura do Corpo humano [°C]

Th = Th + K          #Temperatura do Corpo humano [K]

nTEGs = 12           #Número de TEGs em //

Rmteg = 3.8/nTEGs    #Resistência do conjunto de TEGs [ohm]

Smteg = 0.095        #Coeficiente de Seeback do TEG [V.K^-1]

Pap = 0.00171        #Potência Cosumida pelo aparelho [W]

CargaMax = 51.84     #Carga Máxima da Bateria/Cap [J]  (Bateria rec: 51.84J) (Bateria Cap: 17.1072J)

Eff_bateria = 0.75    #Eficiência de Carregamento da bateria

Preal = []           #List para guardar valores de potência gerada pelo TEG

Pef = []

Energia = [CargaMax] #List para guardar valores de energia  



for n in range(len(temp_month)):

    Tl = temp_month[n+720*(monthToNum(mes)-1)]+K

    deltaT = Th - Tl

    eta_ideal = 1 - math.sqrt(Tl/Th)

    eta_real = eta_ideal*ZT_bite/((1+math.sqrt(1+ZT_bite))**2)

    Voc = Smteg*deltaT

    Preal.append(eta_real*Voc**2/(4*Rmteg))

    Pef.append(Preal[n]-Pap)

    if n%24 >= 7 or n%24 <= 21:        #Horario pessoa acordada

        Energia.append(Energia[n]+(Preal[n]*Eff_bateria-Pap)*60*60)

    elif n%24 < 7 or n%24 > 21:        #Horario pessoa dormindo

        #Energia.append(Energia[n]+(Preal[n])*60*60)  #carrega enquanto dorme

        Energia.append(Energia[n])                    #não carrega enquanto dorme

    if Energia[n+1] > (CargaMax):

        Energia[n+1] = CargaMax

    elif Energia[n+1] < 0:

        Energia[n+1] = 0

    
tick = list(range(31))

ticks = [(n*24) for n in tick]

fig = plt.figure()

ax = fig.add_axes([0, 0, 0.8, 0.8]) # main axes

ax.plot(Energia)

ax.set_xlabel('dia',fontsize=20)

ax.set_ylabel('Carga da Bateria [J]',fontsize=20)

ax.set_title('Carga da Bateria na cidade ' +cidade_drop.value+ ' no mês: '+mes_drop.value, fontsize=28)

ax.set_xticks(ticks)

tick = [n+1 for n in tick]

ax.set_xticklabels(map(str,tick))

plt.show()
tick = list(range(31))

ticks = [(n*24) for n in tick]

fig = plt.figure()

ax = fig.add_axes([0, 0, 0.8, 0.8]) # main axes

Pef_mW = [n*100 for n in Pef]

ax.plot(Pef_mW)

ax.set_xlabel('dia',fontsize=20)

ax.set_ylabel('Potência Eficaz [mW]',fontsize=20)

ax.set_title('Potência Eficaz na Cidade ' +cidade_drop.value+ ' gerada ao longo do mês: '+mes_drop.value, fontsize=28)

ax.set_xticks(ticks)

tick = [n+1 for n in tick]

ax.set_xticklabels(map(str,tick))

plt.show()