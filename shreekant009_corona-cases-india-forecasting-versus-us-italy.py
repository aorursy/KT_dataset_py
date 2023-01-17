import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import folium

import os

from fbprophet import Prophet

import warnings

warnings.filterwarnings('ignore')

from IPython.core.display import HTML

import plotly.graph_objects as go

pd.set_option('display.max_rows',20000, 'display.max_columns',100)
df_carona_in_india = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

df_carona_india = pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")

df_ICMR = pd.read_csv("../input/covid19-in-india/ICMRTestingDetails.csv")

df_Individual = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")

df_Hospital = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")

df_Age = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")

df_Italy = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")

df_daily_US = pd.read_csv("../input/covid19-in-usa/us_states_covid19_daily.csv")

df_daily_USA = pd.read_csv("../input/covid19-in-usa/us_covid19_daily.csv")

df_lab = pd.read_csv("../input/covid19-in-india/ICMRTestingLabs.csv")

df_utm_lab = pd.read_csv("../input/lat-long-lab/City_Lat_Lon.csv")

df_statewise = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")

df_population_of_india = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")

df_with_utm = pd.read_csv("../input/utm-of-india/UTM ZONES of INDIA.csv")

df_carona_in_india.head()
#Total cases of carona in India

df_carona_in_india['Total Cases'] = df_carona_in_india['Cured'] + df_carona_in_india['Deaths'] + df_carona_in_india['Confirmed']

#Active cases of carona in India

df_carona_in_india['Active Cases'] = df_carona_in_india['Total Cases'] - df_carona_in_india['Cured'] - df_carona_in_india['Deaths']

df_carona_in_india.head()
#Till 28th April Cases in India

df1= df_carona_in_india[df_carona_in_india['Date']=='28/04/20']

fig = px.bar(df1, x='State/UnionTerritory', y='Total Cases', color='Total Cases', height=600)

fig.update_layout(

    title='Till 28th April Total Cases in India')

fig.show()
#Till 28th April Active Cases in India

df1= df_carona_in_india[df_carona_in_india['Date']=='28/04/20']

fig = px.bar(df1, x='State/UnionTerritory', y='Active Cases', color='Active Cases',barmode='group', height=600)

fig.update_layout(

    title='Till 28th April Active Cases in India')

fig.show()
df_carona_in_india['Date'] =pd.to_datetime(df_carona_in_india.Date,dayfirst=True)

df_carona_in_india
carona_data = df_carona_in_india.groupby(['Date'])['Active Cases','Cured'].sum().reset_index()

labels = ['Active Cases','Cured']

values = [carona_data['Active Cases'].iloc[-1], carona_data['Cured'].iloc[-1]]



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=15,

                  marker=dict(colors=['#ff3300','#00ff00'], line=dict(color='#FFFFFF', width=2.5)))

fig.update_layout(

    title='COVID-19 ACTIVE CASES VS CURED')

fig.show()

#Daily Cases in India Datewise

carona_data = df_carona_in_india.groupby(['Date'])['Total Cases'].sum().reset_index().sort_values('Total Cases',ascending = True)

carona_data['Daily Cases'] = carona_data['Total Cases'].sub(carona_data['Total Cases'].shift())

carona_data['Daily Cases'].iloc[0] = carona_data['Total Cases'].iloc[0]

carona_data['Daily Cases'] = carona_data['Daily Cases'].astype(int)

fig = px.bar(carona_data, y='Daily Cases', x='Date',hover_data =['Daily Cases'], color='Daily Cases', height=500)

fig.update_layout(

    title='Daily Cases in India Datewise')

fig.show()
HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2157900" data-url="https://flo.uri.sh/visualisation/2157900/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
carona_data['Corona Growth Rate'] = carona_data['Total Cases'].pct_change().mul(100).round(2)

#Corona Growth Rate Comparison with Previous Day

fig = px.bar(carona_data, y='Corona Growth Rate', x='Date',hover_data =['Corona Growth Rate','Total Cases'], height=500)

fig.update_layout(

    title='Corona Growth Rate(in Percentage) Comparison with Previous Day')

fig.show()
#Moratality Rate

carona_data = df_carona_in_india.groupby(['Date'])['Total Cases','Active Cases','Deaths'].sum().reset_index().sort_values('Date',ascending=False)

carona_data['Mortality Rate'] = ((carona_data['Deaths']/carona_data['Total Cases'])*100)

fig = go.Figure()

fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Mortality Rate'],

                    mode='lines+markers',name='Cases',marker_color='red'))

fig.update_layout(title_text='COVID-19 Mortality Rate in INDIA',plot_bgcolor='rgb(225,230,255)')

fig.show()
#DAILY DEATHS IN INDIA

carona_data = df_carona_in_india.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Deaths',ascending = True)

carona_data['DAILY DEATHS'] = carona_data['Deaths'].sub(carona_data['Deaths'].shift())

carona_data['DAILY DEATHS'].iloc[0] = carona_data['Deaths'].iloc[0]

carona_data['DAILY DEATHS'] = carona_data['DAILY DEATHS'].astype(int)

fig = px.bar(carona_data, y='DAILY DEATHS', x='Date',hover_data =['DAILY DEATHS'], color='DAILY DEATHS', height=500)

fig.update_layout(

    title='DAILY DEATHS IN INDIA')

fig.show()
#Recovery Rate

carona_data = df_carona_in_india.groupby(['Date'])['Total Cases','Active Cases','Cured'].sum().reset_index().sort_values('Date',ascending=False)

carona_data['Recovery Rate'] = ((carona_data['Cured']/carona_data['Total Cases'])*100)

fig = go.Figure()

fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Recovery Rate'],

                    mode='lines+markers',name='Cases',marker_color='green'))

fig.update_layout(title_text='COVID-19 Recovery Rate in INDIA',plot_bgcolor='rgb(225,230,255)')

fig.show()
#DAILY Recovery IN INDIA

carona_data = df_carona_in_india.groupby(['Date'])['Cured'].sum().reset_index().sort_values('Cured',ascending = True)

carona_data['DAILY RECOVERY'] = carona_data['Cured'].sub(carona_data['Cured'].shift())

carona_data['DAILY RECOVERY'].iloc[0] = carona_data['Cured'].iloc[0]

carona_data['DAILY RECOVERY'] = carona_data['DAILY RECOVERY'].astype(int)

fig = px.bar(carona_data, y='DAILY RECOVERY', x='Date',hover_data =['DAILY RECOVERY'], color='DAILY RECOVERY', height=500)

fig.update_layout(

    title='DAILY RECOVERY IN INDIA')

fig.show()
#Total Cases in Indian State Datewise

carona_data = df_carona_in_india.groupby(['Date','State/UnionTerritory','Total Cases'])['Cured','Deaths','Active Cases'].sum().reset_index().sort_values('Total Cases',ascending = False)

fig = px.bar(carona_data, y='Total Cases', x='Date',hover_data =['State/UnionTerritory','Active Cases','Deaths','Cured'], color='Total Cases',barmode='group', height=700)

fig.update_layout(

    title='Indian States with Current Total Corona Cases')

fig.show()
df_carona_india.head()
#Pie chart visualization of states effected by caronavirus

fig = px.pie(df_Age, values='TotalCases', names='AgeGroup')

fig.update_layout(

    title='Age Group affected with COVID-19')

fig.show()
#Agewise Gender affected by COVID-19(There are lot of Nan Values)

fig = px.histogram(df_Individual.dropna(), x="age",color ='gender')

fig.update_layout(

    title='Agewise Gender affected by COVID-19')

fig.show()
#Agewise Covid-19 patients in State(There are lot of Nan Values)

fig = px.histogram(df_Individual.dropna(), x="age",color ='detected_state')

fig.update_layout(

    title='Agewise Covid-19 patients in State')

fig.show()
#Statewise Total Cases

df_new = df1.groupby(['State/UnionTerritory','Cured','Deaths','Active Cases'])['Total Cases'].sum().reset_index().sort_values('Total Cases',ascending = False)



#Using Merge two join two diffrent data frames and then sorting them in ascending

df_population_with_carona_case = df_population_of_india.merge(df_new, left_on='State / Union Territory', right_on='State/UnionTerritory')

df_population_with_carona_case=df_population_with_carona_case.drop(labels=['State/UnionTerritory','Sno'],axis=1)

df_population_with_carona_case=df_population_with_carona_case.sort_values('Total Cases',ascending=False)

df_population_with_carona_case
#Pie chart visualization of states effected by caronavirus

fig = px.pie(df_population_with_carona_case, values='Total Cases', names='State / Union Territory')

fig.update_layout(

    title='Pie chart visualization of states effected by caronavirus')

fig.show()
#Data of Population,Cases of Carona and UTM 

df_pop_caro_utm = df_population_with_carona_case.merge(df_with_utm , left_on='State / Union Territory', right_on='State / Union Territory')

df_pop_caro_utm

#Active Case in Indian States

fig = go.Figure(data=[go.Scatter(

    x=df_pop_caro_utm['State / Union Territory'][0:36],

    y=df_pop_caro_utm['Total Cases'][0:36],

    mode='markers',

    marker=dict(

        size=[100,90,80, 70, 60, 50, 40,35,35,35,35,35,35,35,35,35,35,30,28,28,25,25,20,15,15,15,15,10,10,10,10,10,10],

        showscale=True

        )

)])

fig.update_layout(

    title='Total Case in Indian States',

    xaxis_title="States",

    yaxis_title="Total Cases",

)

fig.show()
#Active Case in Indian States

fig = go.Figure(data=[go.Scatter(

    x=df_pop_caro_utm['State / Union Territory'][0:36],

    y=df_pop_caro_utm['Active Cases'][0:36],

    mode='markers',

    marker=dict(

        size=[100,90,80, 70, 60, 50, 40,35,35,35,35,35,35,35,35,35,35,30,28,28,25,25,20,15,15,15,15,10,10,10,10,10,10],

    color=[50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50],

        showscale=True

        )

)])

fig.update_layout(

    title='Active Case in Indian States',

    xaxis_title="States",

    yaxis_title="Active Cases",

)

fig.show()
df_Hospital.at[0,'State/UT'] = 'Andaman and Nicobar Islands'

df_hospital_beds = df_pop_caro_utm.merge(df_Hospital , left_on='State / Union Territory', right_on='State/UT')

df_hospital_beds['NumSubDistrictHospitals_HMIS'] = df_hospital_beds['NumSubDistrictHospitals_HMIS'].fillna(value=0.0)

df_hospital_beds = df_hospital_beds.drop(['Area','Density','Gender Ratio','Latitude','Longitude','Sno','State/UT'],axis=1)

df_hospital_beds['Total Beds'] = df_hospital_beds['NumRuralBeds_NHP18'] + df_hospital_beds['NumUrbanBeds_NHP18']

df_hospital_beds['Total Used Beds'] = df_hospital_beds['Active Cases']

df_hospital_beds['Total Unused Beds'] = df_hospital_beds['Total Beds'] - df_hospital_beds['Active Cases']

df_hospital_beds.head()
#Statewise COVID-19 Beds in INDIA

Beds = df_hospital_beds.groupby(['State / Union Territory'])['Total Beds','Total Unused Beds','Total Used Beds'].sum().reset_index().sort_values('State / Union Territory',ascending =True)



fig = go.Figure()

fig.add_trace(go.Bar(

    y=Beds['State / Union Territory'],

    x=Beds['Total Unused Beds'],

    name='Total Unused Beds',

    orientation='h',

        marker=dict(

        color='rgba(1000, 71, 80, 0.6)',

        line=dict(color='rgba(1000, 71, 80, 1.0)', width=3)

    )



))

fig.add_trace(go.Bar(

    y=Beds['State / Union Territory'],

    x=Beds['Total Used Beds'],

    name='Total Used Beds',

    orientation='h',

    marker=dict(

        color='rgba(100, 78, 139, 0.6)',

        line=dict(color='rgba(100, 78, 139, 1.0)', width=3)

    )    

))

fig.update_layout(

    title='BEDS for Covid19 Patients')

fig.update_layout(barmode='stack')

fig.show()
fig = px.treemap(df_hospital_beds, path=['State / Union Territory'],hover_data=['State / Union Territory','Rural population','NumRuralHospitals_NHP18','NumRuralBeds_NHP18'], values='NumRuralHospitals_NHP18')

fig.update_layout(

    title='Statewise Rural Population and their Hospital Facilities for Covid19')

fig.show()
fig = px.treemap(df_hospital_beds, path=['State / Union Territory'],hover_data=['State / Union Territory','Urban population','NumUrbanHospitals_NHP18','NumUrbanBeds_NHP18'], values='NumUrbanHospitals_NHP18')

fig.update_layout(

    title='Statewise Urban Population and their Hospital Facilities for Covid19')

fig.show()
#India's Map with Statewise data of Total Cases,Deaths and Cure

India_map = folium.Map(location=[20.5937, 78.9629],zoom_start=4.55)

fg=folium.FeatureGroup(name="my map")

fg.add_child(folium.GeoJson(data=(open('../input/states-of-india/states_of_india.json','r',encoding='utf-8-sig').read())))

India_map.add_child(fg)

for lat,lan,name,cured,deaths,cases in zip(df_pop_caro_utm['Latitude'],df_pop_caro_utm['Longitude'],df_pop_caro_utm['State / Union Territory'],df_pop_caro_utm['Cured'],df_pop_caro_utm['Deaths'],df_pop_caro_utm['Total Cases']):

    if(deaths == 0):

        folium.Marker(location=[lat,lan],popup="<b>State  : </b>"+name+ "<br> <b>Total Cases : </b> "+str(cases)+"<br> <b>Deaths : </b> "+str(deaths)+"<br> <b>Cured : </b> "+str(cured)).add_to(India_map)

    else:

        folium.Marker(location=[lat,lan],popup="<b>State  : </b>"+name+ "<br> <b>Total Cases : </b> "+str(cases)+"<br> <b>Deaths : </b> "+str(deaths)+"<br> <b>Cured : </b> "+str(cured),icon=folium.Icon(color="red")).add_to(India_map)

India_map
#Total Cases,Active Cases,Cured,Deaths from Corona Virus in India

carona_data = df_carona_in_india.groupby(['Date'])['Total Cases','Active Cases','Cured','Deaths'].sum().reset_index().sort_values('Date',ascending=False)

fig = go.Figure()

fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Total Cases'],

                    mode='lines+markers',name='Total Cases'))

fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Active Cases'], 

                mode='lines+markers',name='Active Cases'))

fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Cured'], 

                mode='lines+markers',name='Cured'))

fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Deaths'], 

                mode='lines+markers',name='Deaths'))

fig.update_layout(title_text='Curve Showing Different Cases from COVID-19 in India',plot_bgcolor='rgb(225,230,255)')

fig.show()
#Covid Centers in India

df_lab_utm = pd.concat([df_lab, df_utm_lab], axis=1, join='inner')

India_lab_map = folium.Map(location=[25, 80], zoom_start=5)



for lat,lan,lab,address,labtype in zip(df_lab_utm['Latitude'],df_lab_utm['Longitude'],df_lab_utm['lab'],df_lab_utm['address'],df_lab_utm['type']):

    if(labtype == 'Government Laboratory'):

        folium.Marker(location=[lat,lan],popup="<b>Center Name  : </b>"+lab+ "<br> <b>Address : </b> "+str(address)+"<br> <b>Labs : </b> "+str(labtype),icon=folium.Icon(color="red")).add_to(India_lab_map)

    

    elif(labtype == 'Private Laboratory'):

        folium.Marker(location=[lat,lan],popup="<b>Center Name  : </b>"+lab+ "<br> <b>Address : </b> "+str(address)+"<br> <b>Labs : </b> "+str(labtype),icon=folium.Icon(color="green")).add_to(India_lab_map)

    

    elif(labtype == 'Collection Site'):

        folium.Marker(location=[lat,lan],popup="<b>Center Name  : </b>"+lab+ "<br> <b>Address : </b> "+str(address)+"<br> <b>Labs : </b> "+str(labtype),icon=folium.Icon(color="blue")).add_to(India_lab_map)                   

        

India_lab_map
#Testing till 19th April

df_ICMR ['DateTime'] =pd.to_datetime(df_ICMR .DateTime,dayfirst=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_ICMR['DateTime'], y=df_ICMR['TotalSamplesTested'],

                    mode='lines+markers',name='TotalSamplesTested'))

fig.add_trace(go.Scatter(x=df_ICMR['DateTime'], y=df_ICMR['TotalIndividualsTested'], 

                mode='lines+markers',name='TotalIndividualsTested'))

fig.add_trace(go.Scatter(x=df_ICMR['DateTime'], y=df_ICMR['TotalPositiveCases'], 

                mode='lines+markers',name='TotalPositiveCases'))

fig.update_layout(title_text='ICMR TEST for COVID-19',plot_bgcolor='rgb(225,230,255)')

fig.show()
#ICMR APPROVED LABS

fig = px.histogram(df_lab, x="state",color ='type')

fig.update_layout(

    title='ICMR APPROVED LABS')

fig.show()
#Statewise COVID-19 Testing in INDIA

carona_data = df_statewise.groupby(['State'])['TotalSamples','Negative','Positive'].sum().reset_index().sort_values('State',ascending =True)

fig = go.Figure()

fig.add_trace(go.Bar(

    y=carona_data['State'],

    x=carona_data['Negative'],

    name='Negative',

    orientation='h',

        marker=dict(

        color='rgba(58, 71, 80, 0.6)',

        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)

    )



))

fig.add_trace(go.Bar(

    y=carona_data['State'],

    x=carona_data['Positive'],

    name='Positive',

    orientation='h',

    marker=dict(

        color='rgba(246, 78, 139, 0.6)',

        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)

    )    

))

fig.update_layout(

    title='Statewise COVID Tests')

fig.update_layout(barmode='stack')

fig.show()
#Forecasting of Total Cases for Next 30 Days

df = df_carona_in_india.groupby('Date')['Total Cases'].sum().reset_index()

df.columns = ['ds','y']

df['ds'] = pd.to_datetime(df['ds'])

m = Prophet()

m.fit(df)

future = m.make_future_dataframe(periods=30)

forecast = m.predict(future)

fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],

                    mode='lines+markers',name='Predicted Cases',marker_color='Black'))

fig.add_trace(go.Scatter(x=df['ds'], y=df['y'],

                    mode='lines+markers',name='Actual Cases',marker_color='red'))

fig.update_layout(

    title='Forecasting of Total Cases in INDIA for Next 30 Days')

fig.show()
#Forecasting of Deaths for Next 30 Days

df = df_carona_in_india.groupby('Date')['Deaths'].sum().reset_index()

df.columns = ['ds','y']

df['ds'] = pd.to_datetime(df['ds'])

m = Prophet()

m.fit(df)

future = m.make_future_dataframe(periods=30)

forecast = m.predict(future)

fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],

                    mode='lines+markers',name='Predicted Cases',marker_color='red'))

fig.add_trace(go.Scatter(x=df['ds'], y=df['y'],

                    mode='lines+markers',name='Actual Cases',marker_color='blue'))

fig.update_layout(

    title='Forecasting of Deaths in INDIA for Next 30 Days')

fig.show()
#Forecasting of Cured for Next 30 Days

df = df_carona_in_india.groupby('Date')['Cured'].sum().reset_index()

df.columns = ['ds','y']

df['ds'] = pd.to_datetime(df['ds'])

m = Prophet()

m.fit(df)

future = m.make_future_dataframe(periods=30)

forecast = m.predict(future)

fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],

                    mode='lines+markers',name='Predicted Cases',marker_color='green'))

fig.add_trace(go.Scatter(x=df['ds'], y=df['y'],

                    mode='lines+markers',name='Actual Cases',marker_color='yellow'))

fig.update_layout(

    title='Forecasting of Cured in INDIA for Next 30 Days')

fig.show()
#Total Confirmed Cases INDIA vs ITALY

df_Italy_p = pd.read_csv("../input/covid19-in-italy/covid19_italy_province.csv")

carona_data_ITALY = df_Italy_p.groupby(['Date'])['TotalPositiveCases'].sum().reset_index().sort_values('TotalPositiveCases',ascending = True)



carona_data_INDIA =df_carona_in_india.groupby(['Date'])['Confirmed'].sum().reset_index().sort_values('Date',ascending = True)



fig = go.Figure()

fig.add_trace(go.Scatter(x=carona_data_ITALY['Date'], y=carona_data_ITALY['TotalPositiveCases'],

                    mode='lines+markers',name='Italy'))

fig.add_trace(go.Scatter(x=carona_data_INDIA['Date'], y=carona_data_INDIA['Confirmed'],

                    mode='lines+markers',name='India'))

fig.update_layout(title_text='Total Confirmed Cases INDIA vs ITALY',plot_bgcolor='rgb(225,230,255)')

fig.show()
#COVID-19 Recovered Cases in INDIA Vs ITALY

df_Italy_r = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")

df_Italy_r['Total Cases'] = df_Italy_r['HospitalizedPatients']+df_Italy_r['IntensiveCarePatients'] + df_Italy_r['TotalHospitalizedPatients'] + df_Italy_r['HomeConfinement'] + df_Italy_r['CurrentPositiveCases'] + df_Italy_r['NewPositiveCases'] + df_Italy_r['Recovered'] + df_Italy_r['Deaths'] + df_Italy_r['TotalPositiveCases']

carona_data_ITALY = df_Italy_r.groupby(['Date'])['Recovered'].sum().reset_index().sort_values('Date',ascending = True)

df_carona_in_india['Date'] =pd.to_datetime(df_carona_in_india.Date,dayfirst=True)

carona_data_INDIA =df_carona_in_india.groupby(['Date'])['Cured'].sum().reset_index().sort_values('Date',ascending = True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=carona_data_ITALY['Date'], y=carona_data_ITALY['Recovered'],

                    mode='lines+markers',name='Italy'))

fig.add_trace(go.Scatter(x=carona_data_INDIA['Date'], y=carona_data_INDIA['Cured'],

                    mode='lines+markers',name='India'))

fig.update_layout(title_text='COVID-19 Recovered Cases in INDIA Vs ITALY',plot_bgcolor='rgb(225,230,255)')

fig.show()
#COVID-19 Death Cases in ITALY VS INDIA

df_Italy_r = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")

df_Italy_r['Total Cases'] = df_Italy_r['HospitalizedPatients']+df_Italy_r['IntensiveCarePatients'] + df_Italy_r['TotalHospitalizedPatients'] + df_Italy_r['HomeConfinement'] + df_Italy_r['CurrentPositiveCases'] + df_Italy_r['NewPositiveCases'] + df_Italy_r['Recovered'] + df_Italy_r['Deaths'] + df_Italy_r['TotalPositiveCases']

carona_data_ITALY = df_Italy_r.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Date',ascending = True)

df_carona_in_india['Date'] =pd.to_datetime(df_carona_in_india.Date,dayfirst=True)

carona_data_INDIA =df_carona_in_india.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Date',ascending = True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=carona_data_ITALY['Date'], y=carona_data_ITALY['Deaths'],

                    mode='lines+markers',name='Italy'))

fig.add_trace(go.Scatter(x=carona_data_INDIA['Date'], y=carona_data_INDIA['Deaths'],

                    mode='lines+markers',name='India'))

fig.update_layout(title_text='COVID-19 Death Cases in INDIA Vs ITALY',plot_bgcolor='rgb(225,230,255)')

fig.show()
#Daily Deaths in Italy

carona_data = df_Italy_r.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Deaths',ascending = True)

carona_data['Daily Deaths'] = carona_data['Deaths'].sub(carona_data['Deaths'].shift())

carona_data['Daily Deaths'].iloc[0] = carona_data['Deaths'].iloc[0]

carona_data['Daily Deaths'] = carona_data['Daily Deaths'].astype(int)

fig = px.bar(carona_data, y='Daily Deaths', x='Date',hover_data =['Daily Deaths'], color='Daily Deaths', height=500)

fig.update_layout(

    title='DAILY DEATHS IN ITALY')

fig.show()
#Total Corona Cases INDIA vs USA

carona_data_INDIA =df_carona_in_india.groupby(['Date'])['Total Cases'].sum().reset_index().sort_values('Date',ascending = True)

carona_data_USA= df_daily_US.groupby(['dateChecked'])['positive'].sum().reset_index().sort_values('dateChecked',ascending = True)

fig = go.Figure()



fig.add_trace(go.Scatter(x=carona_data_USA['dateChecked'], y=carona_data_USA['positive'],

                    mode='lines+markers',name='USA'))



fig.add_trace(go.Scatter(x=carona_data_INDIA['Date'], y=carona_data_INDIA['Total Cases'],

                    mode='lines+markers',name='India'))



fig.update_layout(title_text='Total Corona Cases INDIA vs USA',plot_bgcolor='rgb(225,230,255)')

fig.show()
#COVID-19 Recovered Cases in INDIA vs USA

carona_data_INDIA =df_carona_in_india.groupby(['Date'])['Cured'].sum().reset_index().sort_values('Date',ascending = True)

carona_data_USA= df_daily_US.groupby(['dateChecked'])['recovered'].sum().reset_index().sort_values('dateChecked',ascending = True)

fig = go.Figure()



fig.add_trace(go.Scatter(x=carona_data_USA['dateChecked'], y=carona_data_USA['recovered'],

                    mode='lines+markers',name='USA'))



fig.add_trace(go.Scatter(x=carona_data_INDIA['Date'], y=carona_data_INDIA['Cured'],

                    mode='lines+markers',name='India'))



fig.update_layout(title_text='COVID-19 Recovered Cases in INDIA vs USA',plot_bgcolor='rgb(225,230,255)')

fig.show()
#COVID-19 Death Cases in INDIA vs USA

carona_data_INDIA =df_carona_in_india.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Date',ascending = True)

carona_data_USA= df_daily_US.groupby(['dateChecked'])['death'].sum().reset_index().sort_values('dateChecked',ascending = True)

fig = go.Figure()



fig.add_trace(go.Scatter(x=carona_data_USA['dateChecked'], y=carona_data_USA['death'],

                    mode='lines+markers',name='USA'))



fig.add_trace(go.Scatter(x=carona_data_INDIA['Date'], y=carona_data_INDIA['Deaths'],

                    mode='lines+markers',name='India'))



fig.update_layout(title_text='COVID-19 Death Cases in INDIA vs USA',plot_bgcolor='rgb(225,230,255)')

fig.show()
#DAILY DEATHS IN USA

carona_data = df_daily_US.groupby(['dateChecked'])['death'].sum().reset_index().sort_values('death',ascending = True)

carona_data['DAILY DEATHS'] = carona_data['death'].sub(carona_data['death'].shift())

carona_data['DAILY DEATHS'].iloc[0] = carona_data['death'].iloc[0]

carona_data['DAILY DEATHS'] = carona_data['DAILY DEATHS'].astype(int)

fig = px.bar(carona_data, y='DAILY DEATHS', x='dateChecked',hover_data =['DAILY DEATHS'], color='DAILY DEATHS', height=500)

fig.update_layout(

    title='DAILY DEATHS IN USA')

fig.show()