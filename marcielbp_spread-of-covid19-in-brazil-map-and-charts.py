# Required libraries - data visualization
import numpy as np # linear algebra
import pandas as pd 
import altair as alt
import geopandas as gpd
import json
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
from plotly.offline import init_notebook_mode, plot, iplot, download_plotlyjs
import plotly as ply
import pycountry
import folium 
from folium import plugins
from scipy.optimize import curve_fit


coronaBrasil = pd.read_csv("/kaggle/input/corona-virus-brazil/brazil_covid19.csv")
popBrasil = pd.read_csv("/kaggle/input/geodatabrazil/populacaoEstado.csv")
numEstados = len(popBrasil)
coronaBrasil=coronaBrasil.fillna(0)

delta = timedelta(days=1)
numLookaheadDays = 15
numPastDays = 30
pastDays = timedelta(days=numPastDays)
lookAhead = timedelta(days=numLookaheadDays)
mainStartDate = date.today() - pastDays
mainEndDate = date.today()
mainProjectedDate = date.today() + lookAhead
today = date.today()
if (sum(coronaBrasil[coronaBrasil['date'] == mainEndDate.strftime("%Y-%m-%d")].cases) <1):
    mainEndDate = mainEndDate - delta
    mainProjectedDate = date.today() - delta + lookAhead
casosDia = []
newCases = 0

casosAcumDia = []
newCasesAcum = 0

deathsAcum = []
deathsAcumDia = 0

data = []

start_date = mainStartDate
end_date = mainEndDate

# The sum of all corona new cases for each day
while start_date <= end_date:
    newCasesAcum = sum(coronaBrasil[coronaBrasil['date'] == start_date.strftime("%Y-%m-%d")].cases)
    newCases = newCasesAcum - newCases
    newDeaths = sum(coronaBrasil[coronaBrasil['date'] == start_date.strftime("%Y-%m-%d")].deaths)
    casosDia.append(newCases)
    casosAcumDia.append(newCasesAcum)
    deathsAcum.append(newDeaths)
    data.append(start_date)
    start_date = start_date + delta
fig = go.Figure()

yaux = np.array(casosAcumDia) - np.array(casosDia) - np.array(deathsAcum)

fig.add_trace(go.Bar(y=casosDia,x=data, name="New Cases"))
fig.add_trace(go.Bar(y=deathsAcum,x=data, name="Deaths"))
fig.add_trace(go.Bar(y=yaux,x=data, name="Previous Cases"))
fig.update_layout(barmode='stack',title='Increase of covid19 cases and deaths in Brazil by Date', xaxis_title='Day',plot_bgcolor='white')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
fig.update_yaxes(title_text="Number of Occurrences",showgrid=True, gridwidth=0.5, gridcolor='LightGray')
fig.show()
casosStateDia = []
deathsStateDia = []
newCasesStateDia = 0
newDeathsStateDia = 0

data = []

start_date = mainStartDate
end_date = mainEndDate

fig = go.Figure()
# plot for each state
for stateNumber in range (0,numEstados):
    while start_date <= end_date:
        coronaBrasilCurrent = coronaBrasil[coronaBrasil['state']==popBrasil['estado'][stateNumber]]
        newCasesStateDia = sum(coronaBrasilCurrent[coronaBrasilCurrent['date'] == start_date.strftime("%Y-%m-%d")].cases)
        newDeathsStateDia = sum(coronaBrasilCurrent[coronaBrasilCurrent['date'] == start_date.strftime("%Y-%m-%d")].deaths)
        casosStateDia.append(newCasesStateDia)#cases for 100.000 hab
        deathsStateDia.append(newDeathsStateDia)#cases for 100.000 hab
        data.append(start_date)
        start_date = start_date + delta
    fig.add_trace(go.Scatter(y=casosStateDia,x=data, name=popBrasil['estado'][stateNumber]))
    casosStateDia = []
    deathsStateDia = []
    newCasesStateDia = 0
    newDeathsStateDia = 0
    start_date = mainStartDate
    end_date = mainEndDate
fig.update_layout(title='Spread of Corona Virus in Brazil by Date - States', xaxis_title='Day',plot_bgcolor='white')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
fig.update_yaxes(title_text="Number of Occurrences",showgrid=True, gridwidth=1, gridcolor='LightGray')

fig.show()
casosPerCapitaDia = []
newCasesPerCapitaDia = 0
data = []

start_date = mainStartDate
end_date = mainEndDate

fig = go.Figure()
# plot for each state
for stateNumber in range (0,numEstados):
    while start_date <= end_date:
        coronaBrasilCurrent = coronaBrasil[coronaBrasil['state']==popBrasil['estado'][stateNumber]]
        newCasesPerCapitaDia = sum(coronaBrasilCurrent[coronaBrasilCurrent['date'] == start_date.strftime("%Y-%m-%d")].cases)/popBrasil['população'][stateNumber]
        casosPerCapitaDia.append(newCasesPerCapitaDia*100000)#cases for 100.000 hab
        data.append(start_date)
        start_date = start_date + delta
    fig.add_trace(go.Scatter(y=casosPerCapitaDia,x=data, name=popBrasil['estado'][stateNumber]))
    casosPerCapitaDia = []
    newCasesPerCapitaDia = 0
    start_date = mainStartDate
    end_date = mainEndDate
fig.update_layout(title='Spread of Corona Virus in Brazil by Date - States', xaxis_title='Day',plot_bgcolor='white')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
fig.update_yaxes(title_text="Cases per 100.000 hab",showgrid=True, gridwidth=1, gridcolor='LightGray')
fig.show()
casosDia = []
newCasesPerCapitaDia = 0
deathsDia = []
newDeathsPerCapitaDia = 0
data = []

start_date = mainStartDate
end_date = mainEndDate


fig = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()
# plot for each state
for regionName in ('Norte','Nordeste','Sul','Centro-oeste','Sudeste'):
    stateArray = popBrasil[popBrasil['região']==regionName].estado
    popRegion = popBrasil[popBrasil['região']==regionName]['população'].sum()/100000
    newCasesPerCapitaDia = 0
    newDeathsPerCapitaDia = 0
    while start_date <= end_date:
        coronaBrasilCurrent = []
        for stateName in stateArray:
            coronaBrasilCurrent = coronaBrasil[coronaBrasil['state']==stateName]
            newCasesPerCapitaDia = newCasesPerCapitaDia + sum(coronaBrasilCurrent[coronaBrasilCurrent['date'] == start_date.strftime("%Y-%m-%d")].cases)
            newDeathsPerCapitaDia = newDeathsPerCapitaDia + sum(coronaBrasilCurrent[coronaBrasilCurrent['date'] == start_date.strftime("%Y-%m-%d")].deaths)
        casosDia.append(newCasesPerCapitaDia)
        deathsDia.append(newDeathsPerCapitaDia)
        newCasesPerCapitaDia = 0
        newDeathsPerCapitaDia = 0
        data.append(start_date)
        start_date = start_date + delta
    casosPerCapitaDia = casosDia/popRegion
    deathsPerCapita = deathsDia/popRegion
    fig.add_trace(go.Bar(y=casosPerCapitaDia,x=data, name=regionName))
    fig2.add_trace(go.Scatter(y=casosPerCapitaDia,x=data, name=regionName))
    fig3.add_trace(go.Scatter(y=deathsPerCapita,x=data, name=regionName,mode='lines'))
    casosDia = []
    newCasesPerCapitaDia = 0
    deathsDia = []
    newDeathsPerCapitaDia = 0
    data = []
    start_date = mainStartDate
    end_date = mainEndDate

coronaBrasilCurrent = []
coronaBrasilDeaths = []
popRegion = popBrasil['população'].sum()/100000
newCasesDia = 0
newDeathsDia = 0
while start_date <= end_date:
    newCasesDia = sum(coronaBrasil[coronaBrasil['date'] == start_date.strftime("%Y-%m-%d")].cases)
    newDeathsDia = sum(coronaBrasil[coronaBrasil['date'] == start_date.strftime("%Y-%m-%d")].deaths)
    coronaBrasilCurrent.append(newCasesDia)
    coronaBrasilDeaths.append(newDeathsDia)
    data.append(start_date)
    start_date = start_date + delta
casosPerCapitaDia = coronaBrasilCurrent/popRegion
deathsPerCapita = coronaBrasilDeaths/popRegion
fig.add_trace(go.Scatter(y=casosPerCapitaDia,x=data, name='Brasil',mode='lines+markers'))
fig2.add_trace(go.Scatter(y=casosPerCapitaDia,x=data, name='Brasil',mode='lines+markers'))
fig3.add_trace(go.Scatter(y=deathsPerCapita,x=data, name='Brasil',mode='lines+markers'))

fig.update_layout(title='Spread of Corona Virus in Brazil by Date - Regions', xaxis_title='Day',plot_bgcolor='white')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
fig.update_yaxes(title_text="Cases per 100.000 hab",showgrid=True, gridwidth=1, gridcolor='LightGray')

fig2.update_layout(title='Spread of Corona Virus in Brazil by Date - Regions', xaxis_title='Day',plot_bgcolor='white')
fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
fig2.update_yaxes(title_text="Cases per 100.000 hab",showgrid=True, gridwidth=1, gridcolor='LightGray')

fig3.update_layout(title='Deaths by Corona Virus in Brazil', xaxis_title='Day',plot_bgcolor='white')
fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
fig3.update_yaxes(title_text="Deaths per 100.000 hab",showgrid=True, gridwidth=1, gridcolor='LightGray')

fig.show()
fig2.show()
fig3.show()
covidBR = coronaBrasil.groupby(['state'])['state','cases','deaths'].max()
covidBR['state'] = covidBR.index
covidBR.index = np.arange(1, len(covidBR.state.unique().tolist())+1)
covidBR = covidBR[['state', 'cases','deaths']]
covidBR.rename(columns={'state': 'NOME', 'cases':'cases','deaths':'deaths'}, inplace=True)

covidBR.loc[:,'pop'] = np.zeros(len(covidBR))
for i in range(0,len(popBrasil)):
    # a little complicated but in this line we select lines with current State to set population according to pop dataset
    covidBR.loc[covidBR['NOME'] == popBrasil['estado'][i],'pop']=popBrasil['população'][i]
    
covidBR.loc[:,'casesPerCapita'] = covidBR['cases']/covidBR['pop']
covidBR.loc[:,'casesPer100mil'] = covidBR['casesPerCapita']*100000
covidBR.sort_values(by=['casesPer100mil'], ascending=False).style.background_gradient(cmap='Reds')

listStates = covidBR.sort_values(by=['cases'], ascending=False)[0:3]['NOME'] # 6 estados com maior incidência

casosPerCapitaDia = []
newCasesPerCapitaDia = 0
data = []
dataInt = []

start_date = mainStartDate
end_date = mainEndDate

def func(x, a, b, c):
    return a * np.exp(b * x)

fig = go.Figure()

currentDate = 0
for state in listStates:
    popState = int(popBrasil[popBrasil['estado']==state]['população'])
    while start_date <= end_date:
        
        coronaBrasilCurrent = coronaBrasil[coronaBrasil['state']==state]
        newCasesPerCapitaDia = sum(coronaBrasilCurrent[coronaBrasilCurrent['date'] == start_date.strftime("%Y-%m-%d")].cases)
        casosPerCapitaDia.append(newCasesPerCapitaDia)#cases for 100.000 hab
        data.append(start_date)
        dataInt.append(currentDate)
        currentDate = currentDate + 1
        start_date = start_date + delta
    #fig.add_trace(go.Bar(y=casosPerCapitaDia,x=data, name=state))
    currentDate = 0
    fig.add_trace(go.Scatter(y=casosPerCapitaDia,x=data, name=state))
    #fit curve for recent 7 samples
    popt, pcov = curve_fit(func, dataInt[(len(dataInt)-numLookaheadDays):len(dataInt)], casosPerCapitaDia[(len(casosPerCapitaDia)-numLookaheadDays):len(casosPerCapitaDia)])
    #yy = func(np.array(dataInt), *popt)
    #fig.add_trace(go.Scatter(y=yy,x=data, name=state+ ' (fit)',mode='lines',line = dict(width=2, dash='dash')))
    initialDateInt = max(dataInt)
    casosPerCapitaDia = []
    dataInt = []
    newCasesPerCapitaDia = 0
    start_date = mainEndDate
    projectedDays = []
    projectedDaysInt = []
    valueIndex = initialDateInt
    while start_date <= mainProjectedDate:
        projectedDays.append(start_date)
        projectedDaysInt.append(valueIndex)
        start_date = start_date + delta
        valueIndex = valueIndex +1
    yy = func(np.array(projectedDaysInt), *popt)
    fig.add_trace(go.Scatter(y=yy,x=projectedDays, name=state+ ' (projected)',mode='lines',
                         line = dict(width=2, dash='dash')))
    start_date = mainStartDate
    end_date = mainEndDate

fig.update_layout(title='covid19 BR - 1st-3rd States with most number of cases', xaxis_title='Day',plot_bgcolor='white')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
fig.update_yaxes(title_text="Occurrences",showgrid=True, gridwidth=1, gridcolor='LightGray')

fig.show()
listStates = covidBR.sort_values(by=['cases'], ascending=False)[3:8]['NOME'] # 6 estados com maior incidência

casosPerCapitaDia = []
newCasesPerCapitaDia = 0
data = []
dataInt = []

start_date = mainStartDate
end_date = mainEndDate

def func(x, a, b, c):
    return a * np.exp(b * x)

fig = go.Figure()

currentDate = 0
for state in listStates:
    popState = int(popBrasil[popBrasil['estado']==state]['população'])
    while start_date <= end_date:
        
        coronaBrasilCurrent = coronaBrasil[coronaBrasil['state']==state]
        newCasesPerCapitaDia = sum(coronaBrasilCurrent[coronaBrasilCurrent['date'] == start_date.strftime("%Y-%m-%d")].cases)
        casosPerCapitaDia.append(newCasesPerCapitaDia)#cases for 100.000 hab
        data.append(start_date)
        dataInt.append(currentDate)
        currentDate = currentDate + 1
        start_date = start_date + delta
    #fig.add_trace(go.Bar(y=casosPerCapitaDia,x=data, name=state))
    currentDate = 0
    fig.add_trace(go.Scatter(y=casosPerCapitaDia,x=data, name=state))
    popt, pcov = curve_fit(func, dataInt[(len(dataInt)-numLookaheadDays):len(dataInt)], casosPerCapitaDia[(len(casosPerCapitaDia)-numLookaheadDays):len(casosPerCapitaDia)])
    #yy = func(np.array(dataInt), *popt)
    #fig.add_trace(go.Scatter(y=yy,x=data, name=state+ ' (fit)',mode='lines',line = dict(width=2, dash='dash')))
    initialDateInt = max(dataInt)
    casosPerCapitaDia = []
    dataInt = []
    newCasesPerCapitaDia = 0
    start_date = mainEndDate
    projectedDays = []
    projectedDaysInt = []
    valueIndex = initialDateInt
    while start_date <= mainProjectedDate:
        projectedDays.append(start_date)
        projectedDaysInt.append(valueIndex)
        start_date = start_date + delta
        valueIndex = valueIndex +1
    yy = func(np.array(projectedDaysInt), *popt)
    fig.add_trace(go.Scatter(y=yy,x=projectedDays, name=state+ ' (projected)',mode='lines',line = dict(width=2, dash='dash')))
    start_date = mainStartDate
    end_date = mainEndDate

fig.update_layout(title='covid19 BR - 4th-8th States with most number of confirmed cases', xaxis_title='Day',plot_bgcolor='white')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
fig.update_yaxes(title_text="Occurrences",showgrid=True, gridwidth=1, gridcolor='LightGray')

fig.show()
covidBR.sort_values(by=['deaths'], ascending=False)[0:5].style.background_gradient(cmap='Reds')
listStates = covidBR.sort_values(by=['deaths'], ascending=False)[0:5]['NOME'] # 5 estados com maior numero de mortos
casosPerCapitaDia = []
newCasesPerCapitaDia = 0
data = []
dataInt = []

start_date = mainStartDate
end_date = mainEndDate

def func(x, a, b, c):
    return a * np.exp(b * x)

fig = go.Figure()

currentDate = 0
for state in listStates:
    popState = int(popBrasil[popBrasil['estado']==state]['população'])
    while start_date <= end_date:
        
        coronaBrasilCurrent = coronaBrasil[coronaBrasil['state']==state]
        newCasesPerCapitaDia = sum(coronaBrasilCurrent[coronaBrasilCurrent['date'] == start_date.strftime("%Y-%m-%d")].deaths)
        casosPerCapitaDia.append(newCasesPerCapitaDia)#cases for 100.000 hab
        data.append(start_date)
        dataInt.append(currentDate)
        currentDate = currentDate + 1
        start_date = start_date + delta
    #fig.add_trace(go.Bar(y=casosPerCapitaDia,x=data, name=state))
    
    fig.add_trace(go.Scatter(y=casosPerCapitaDia,x=data, name=state))
    popt, pcov = curve_fit(func, dataInt[(len(dataInt)-numLookaheadDays):len(dataInt)], casosPerCapitaDia[(len(casosPerCapitaDia)-numLookaheadDays):len(casosPerCapitaDia)])
    #yy = func(np.array(dataInt), *popt)
    #fig.add_trace(go.Scatter(y=yy,x=data, name=state+ ' (fit)',mode='lines',line = dict(width=2, dash='dash')))
    initialDateInt = max(dataInt)
    start_date = mainEndDate
    projectedDays = []
    projectedDaysInt = []
    valueIndex = initialDateInt
    # start_date = start_date + delta
    while start_date <= mainProjectedDate:
        projectedDays.append(start_date)
        projectedDaysInt.append(valueIndex)
        start_date = start_date + delta
        valueIndex = valueIndex +1
    yy = func(np.array(projectedDaysInt), *popt)
    fig.add_trace(go.Scatter(y=yy,x=projectedDays, name=state+ ' (projected)',mode='lines',line = dict(width=2, dash='dash')))
    start_date = mainStartDate
    end_date = mainEndDate
    currentDate = 0
    casosPerCapitaDia = []
    newCasesPerCapitaDia = 0
    data = []
    dataInt = []

fig.update_layout(title='covid19 BR - 5 states with most number of deaths', xaxis_title='Day',plot_bgcolor='white')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
fig.update_yaxes(title_text="Occurences",showgrid=True, gridwidth=1, gridcolor='LightGray')

fig.show()
br_lat = -20.000
br_lon = -60.000
with open('/kaggle/input/geodatabrazil/estados.json') as file:
    estadosBR = json.load(file)
covidBR = coronaBrasil.groupby(['state'])['state', 'cases','deaths'].max()
covidBR['state'] = covidBR.index
covidBR.index = np.arange(1, len(covidBR.state.unique().tolist())+1)
covidBR = covidBR[['state', 'cases','deaths']]
covidBR.rename(columns={'state': 'NOME','cases':'cases','deaths':'deaths'}, inplace=True)
covidBR.sort_values(by=['cases'], ascending=False).style.background_gradient(cmap='Reds')
brazil_conf_choropleth = go.Figure(go.Choroplethmapbox(geojson=estadosBR, locations=covidBR['NOME'],
                                                      z=covidBR['cases'], featureidkey="properties.NOME", colorscale='YlGnBu',marker_opacity=0.85, marker_line_width=0.25,text="casos"))

brazil_conf_choropleth.update_layout(mapbox_style="open-street-map", mapbox_zoom=2, 
                                    mapbox_center = {"lat": br_lat, "lon": br_lon})

brazil_conf_choropleth.update_layout(title='Spread of Corona Virus in Brazil by Date - States - per 1000000 hab',margin={"r":0,"t":0,"l":0,"b":0})
iplot(brazil_conf_choropleth)
covidBR.loc[:,'pop'] = np.zeros(len(covidBR))
for i in range(0,len(popBrasil)):
    # a little complicated but in this line we select lines with current State to set population according to pop dataset
    covidBR.loc[covidBR['NOME'] == popBrasil['estado'][i],'pop']=popBrasil['população'][i]
    
covidBR.loc[:,'casesPerCapita'] = covidBR['cases']/covidBR['pop']
covidBR.loc[:,'casesPer100mil'] = covidBR['casesPerCapita']*100000
covidBR.sort_values(by=['casesPer100mil'], ascending=False).style.background_gradient(cmap='Reds')
brazil_conf_choropleth = go.Figure(go.Choroplethmapbox(geojson=estadosBR, locations=covidBR['NOME'],
                                                      z=covidBR['casesPer100mil'], featureidkey="properties.NOME", colorscale='YlGnBu', marker_opacity=0.85, marker_line_width=0.25,text="casos/100 mil"))

brazil_conf_choropleth.update_layout(mapbox_style="open-street-map", mapbox_zoom=2, 
                                    mapbox_center = {"lat": br_lat, "lon": br_lon})

brazil_conf_choropleth.update_layout(title='Spread of Corona Virus in Brazil by Date - States - per 1000000 hab',margin={"r":0,"t":0,"l":0,"b":0})
iplot(brazil_conf_choropleth)