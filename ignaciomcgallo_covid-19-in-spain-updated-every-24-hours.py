import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import geopandas as gpd

import os
import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("https://covid19.isciii.es/resources/serie_historica_acumulados.csv", delimiter=",", encoding="latin1")
data.head()
data.tail(10)
data=data[~data.CCAA.str.startswith('NOTA') & ~data.CCAA.str.startswith('Los') & ~data.CCAA.str.startswith('*') & ~data.CCAA.str.startswith('**') & ~data.CCAA.str.startswith('***') & ~data.CCAA.str.startswith('****')]
data.tail()
data.fillna(0)
data.rename(columns={"FECHA":"Date",

              "PCR+":"Infected",

             "Hospitalizados":"Hospitalized",

             "Fallecidos":"Deaths",

             "Recuperados":"Cured",

             "CASOS":"Cases"},inplace= True)
data.replace({"AN":"Andalucía","AR":"Aragón","AS":"Asturias",

                "IB":"Baleares","CN":"Canarias","CB":"Cantabria",

                 "CM":"Castilla La Mancha","CL":"Castilla y León","CT":"Cataluña",

              "CE":"Ceuta","VC":"C. Valenciana","EX":"Extremadura","GA":"Galicia",

             "MD":"Madrid","ML":"Melilla","MC":"Murcia","NC":"Navarra",

             "PV":"País Vasco","RI":"La Rioja"},inplace=True)
data.describe()
data.isnull().sum()
data.CCAA.unique()
data.Date = pd.to_datetime(data.Date, format="%d/%m/%Y")
data["Time"] = data.Date.apply(lambda x: x.strftime("%d %b, %Y"))
print(data.Date.min())

print(data.Date.max())
total_s = data.groupby(["Date","Time"])["Date","Cases","Infected","TestAc+","Deaths","Cured","Hospitalized","UCI"].sum().reset_index()

total_s.head()
aux = total_s.Infected.to_list()



daily=[]



for i in range(len(aux)-1):

    b = aux[i+1] - aux[i]

    daily.append(b)

    

daily.insert(0,0)   



total_s["Daily_Infected"] = daily
aux = total_s.Deaths.to_list()



daily=[]



for i in range(len(aux)-1):

    b = aux[i+1] - aux[i]

    daily.append(b)

    

daily.insert(0,0)   



total_s["Daily_Deaths"] = daily
aux = total_s.Cases.to_list()



daily=[]



for i in range(len(aux)-1):

    b = aux[i+1] - aux[i]

    daily.append(b)

    

daily.insert(0,0)   



total_s["Daily_Cases"] = daily
total_s.head()
aux = total_s.melt(id_vars="Date", value_vars=("Cases","Infected","TestAc+","Deaths","Cured","UCI","Hospitalized"), value_name="Count" , var_name= "Description")
aux.head()
fig = px.line (aux, x= "Date", y = "Count", color="Description", title= "Actual situation in Spain")

fig.show()
fig = px.line (total_s, x= "Date", y = "Cases", range_x=[data.Date.min(),"2020-04-14"], title= "Total infected in Spain till 14 April, 2020")

fig.show()
fig = px.area (total_s, x= "Date", y = "Daily_Cases", range_x=[data.Date.min(),"2020-04-14"], range_y= [0,total_s.Daily_Cases.max()+10],title= "Daily infections in Spain till 14 April, 2020")

fig.show()
fig = px.area (total_s, x= "Date", y = "Daily_Infected", range_x=["2020-04-15",data.Date.max(),], title= "Daily infections in Spain after 14 April, 2020")

fig.show()
aux = total_s.melt(id_vars="Date", value_vars=("Daily_Deaths","Daily_Cases"), value_name="Count" , var_name= "Description")
aux_i = total_s.melt(id_vars="Date", value_vars=("Daily_Deaths","Daily_Infected"), value_name="Count" , var_name= "Description")
fig = px.area (aux, x= "Date", y = "Count", range_x= [data.Date.min(),"2020-04-14"], range_y=[0,data.Cases.max()], color="Description", title= "Daily infections and deaths till 14 April, 2020")

fig.show()
fig = px.area (aux_i, x= "Date", y = "Count", range_x= ["2020-04-15",data.Date.max()], range_y=[0,data.Infected.max()], color="Description", title= "Daily infections and deaths after 14 April, 2020")

fig.show()
fig = px.bar (aux, x= "Date", y = "Count", color="Description", range_x= [data.Date.min(),"2020-04-14"],range_y=[0,7000], title= "Daily infections and deaths till 14 April, 2020")

fig.show()
fig = px.bar (aux_i, x= "Date", y = "Count", color="Description",  range_x= ["2020-04-15",data.Date.max()], range_y=[0,data.Infected.max()], title= "Daily infections and deaths after 14 April, 2020")

fig.show()
fig = px.line (total_s, x= "Date", y = "Deaths", title= "Total deaths in Spain", color_discrete_sequence = ['red'])

fig.show()
fig = px.area(total_s, x= "Date", y = "Daily_Deaths", title= "Daily deaths in Spain", color_discrete_sequence = ['red'])

fig.show()
data_cases = data[data.Date<="14-04-2020"]
fig = px.bar(data_cases, x="CCAA", y="Cases", color="CCAA",

              animation_frame="Time", animation_group="CCAA", range_y=[0,data.Cases.max()+1000],title= "Infections by regions over time till 14 April,2020")

fig.show()
data_infected = data[data.Date>"14-04-2020"]
fig = px.bar(data_infected, x="CCAA", y="Infected", color="CCAA",

              animation_frame="Time", animation_group="CCAA", range_y=[0,data.Infected.max()+1000],title= "Infections by regions over time")

fig.show()
total_madrid = data[data.CCAA=="Madrid"].groupby("Date")["Date","Infected","Deaths","Cured","Hospitalized","UCI"].sum().reset_index()
aux_m = total_madrid.melt(id_vars="Date", value_vars=("Infected","Deaths","Cured","UCI","Hospitalized"), value_name="Count" , var_name= "Status")
fig = px.line (aux_m, x= "Date", y = "Count", color="Status", title= "Actual situation in Madrid")

fig.show()
fig = px.bar(aux_m, x= "Date", y = "Count", color="Status", title= "Actual situation in Madrid")

fig.show()
total_com = data.groupby(["Date","CCAA"])["Infected","Deaths","Cured","Hospitalized","UCI"].sum().reset_index()

total_com.head()
com = total_com[total_com.Date == max(total_com.Date)]
fig = px.bar (com.sort_values("Infected",ascending=False), x= "CCAA", 

              y = "Infected", color="CCAA", title= "Infections by region", text='CCAA', height=1000, orientation="v")

fig.show()
fig = px.bar (com.sort_values("Deaths",ascending=False), x= "CCAA", 

              y = "Deaths", color="CCAA", title= "Deaths by region", text='CCAA', height=1000, orientation="v")

fig.show()
fig = px.bar (com.sort_values("Hospitalized",ascending=False), x= "CCAA", 

              y = "Hospitalized", color="CCAA", title= "Hospitalizations by region", text='CCAA', height=1000, orientation="v")

fig.show()
fig = px.bar (com.sort_values("UCI",ascending=False), x= "CCAA", 

              y = "UCI", color="CCAA", title= "Advanced care by region", text='CCAA', height=1000, orientation="v")

fig.show()
fig = px.bar (com.sort_values("Cured",ascending=False), x= "CCAA", 

              y = "Cured", color="CCAA", title= "Cured people by region", text='CCAA', height=1000, orientation="v")

fig.show()
aux_all = data.melt(id_vars=["Date", "CCAA"], value_vars=("Cases","Infected","Deaths","Cured","UCI","Hospitalized"), value_name="Count" , var_name= "Status")
aux_all_i = aux_all[aux_all["Status"]=="Cases"]

fig = px.line(aux_all_i,x="Date",y="Count",color="CCAA",range_x=["2020-02-20","2020-04-14"],title="Total infections by region til 14 April,2020")

fig.show()
aux_all_i = aux_all[aux_all["Status"]=="Infected"]

fig = px.line(aux_all_i,x="Date",y="Count",color="CCAA", range_x=["2020-04-15",data.Date.max()], title="Total infections by region")

fig.show()
aux_all_d = aux_all[aux_all["Status"]=="Deaths"]

fig = px.line(aux_all_d,x="Date",y="Count",color="CCAA", title="Total fatalities by region")

fig.show()
aux_all_c = aux_all[aux_all["Status"]=="Cured"]

fig = px.line(aux_all_c,x="Date",y="Count",color="CCAA", title="Total recoveries by region")

fig.show()
for i in data.CCAA.unique(): 

    

    a = i.replace(".","")

    a = a.replace(" ","_")

    

    exec('df_{}=data[data.CCAA == i]'.format(a))

    

    exec('aux_a = df_{}.Infected.to_list()'.format(a))

    

    

    daily=[]

    for i in range(len(aux_a)-1):

        b = aux_a[i+1] - aux_a[i]

        daily.append(b)

    

    daily.insert(0,0)   



    exec('df_{}["Daily_infected"] = daily'.format(a))

    

    exec('aux_d = df_{}.Deaths.to_list()'.format(a))

    

    

    daily=[]

    for i in range(len(aux_d)-1):

        b = aux_d[i+1] - aux_d[i]

        daily.append(b)

    

    daily.insert(0,0)   



    exec('df_{}["Daily_deaths"] = daily'.format(a))
df_daily_infected = pd.DataFrame({"Date":data.Date.unique(),

                                 "Madrid":df_Madrid["Daily_infected"].values,

                                 "Cataluña":df_Cataluña["Daily_infected"].values,

                                 "Andalucia":df_Andalucía["Daily_infected"].values,

                                 "Castilla La Mancha":df_Castilla_La_Mancha["Daily_infected"].values,

                                 "Castilla y Leon":df_Castilla_y_León["Daily_infected"].values,

                                 "País Vasco":df_País_Vasco["Daily_infected"].values})
aux_i = df_daily_infected.melt(id_vars="Date", value_vars=("Madrid","Cataluña","Andalucia","Castilla La Mancha","Castilla y Leon","País Vasco"), value_name="Count" , var_name= "CCAA")
aux_1=aux_i[aux_i.Date>"18-04-2020"]
fig = px.line (aux_1, x= "Date", y = "Count", color="CCAA", title= "Daily infections in Spain (Top 6)")

fig.show()
# In the last days, Cataluña has been the region with a higher number of new cases. This will also be related to the daily fatalities.
fig = px.bar (aux_1, x= "Date", y = "Count", color="CCAA", title= "Daily infections in Spain (Top 6)")

fig.show()
df_daily_fatalities = pd.DataFrame({"Date":data.Date.unique(),

                                 "Madrid":df_Madrid["Daily_deaths"].values,

                                 "Cataluña":df_Cataluña["Daily_deaths"].values,

                                 "Valencia":df_C_Valenciana["Daily_deaths"].values,

                                 "Castilla La Mancha":df_Castilla_La_Mancha["Daily_deaths"].values,

                                 "Castilla y Leon":df_Castilla_y_León["Daily_deaths"].values,

                                 "País Vasco":df_País_Vasco["Daily_deaths"].values})
aux_f = df_daily_fatalities.melt(id_vars="Date", value_vars=("Madrid","Cataluña","Valencia","Castilla La Mancha","Castilla y Leon","País Vasco"), value_name="Count" , var_name= "CCAA")
fig = px.line (aux_f, x= "Date", y = "Count", color="CCAA", title= "Daily fatalities in Spain (Top 6)")

fig.show()
fig = px.bar (aux_f, x= "Date", y = "Count", color="CCAA", title= "Daily fatalities in Spain (Top 6)")

fig.show()
json = "../input/geojson/shapefiles_ccaa_espana.geojson"

geo = gpd.read_file(json)
data.CCAA.unique()
geo.rename(columns={"name_0":"Country",

                        "name_1":"CCAA"},inplace= True)
geo.drop(columns=["id_0","varname_1","nl_name_1","cc_1","type_1","engtype_1","validfr_1","validto_1","remarks_1",

                      "cartodb_id","created_at","updated_at"], inplace=True)
geo.replace({"Castilla-La Mancha":"Castilla La Mancha","Islas Baleares":"Baleares","Islas Canarias":"Canarias",

                "Principado de Asturias":"Asturias","Región de Murcia":"Murcia","Ceuta y Melilla":"Ceuta",

                 "Comunidad de Madrid":"Madrid","Comunidad Foral de Navarra":"Navarra","Comunidad Valenciana":"C. Valenciana"},inplace=True)
mapa=geo.merge(com,on="CCAA",how="left")
mapa.head()
mapa["Time"] = mapa.Date.apply(lambda x: x.strftime("%d %b, %Y"))
import json



with open("../input/geojson/shapefiles_ccaa_espana.geojson") as f:

    geo = json.load(f)
fig = px.choropleth_mapbox(mapa, geojson=geo, locations='id_1', 

                           color='Infected',                           

                           featureidkey="properties.id_1",

                           mapbox_style="carto-positron",

                           zoom=3, center={"lat": 40.4167, "lon": -3.70325},

                           labels={'CCAA':'Infected'},

                           hover_name="CCAA",

                           title="Infections by regional governments"

                          )

#fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth_mapbox(mapa, geojson=geo, locations='id_1', 

                           color='Deaths',                           

                           featureidkey="properties.id_1",

                           mapbox_style="carto-positron",

                           zoom=3, center={"lat": 40.4167, "lon": -3.70325},

                           labels={'CCAA':'Deaths'},

                           hover_name="CCAA",

                           title="Fatalities by regional governments"

                          )

#fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth_mapbox(mapa, geojson=geo, locations='id_1', 

                           color='Hospitalized',                           

                           featureidkey="properties.id_1",

                           mapbox_style="carto-positron",

                           zoom=3, center={"lat": 40.4167, "lon": -3.70325},

                           labels={'CCAA':'Deaths'},

                           hover_name="CCAA"

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()