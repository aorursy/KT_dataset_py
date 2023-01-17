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
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
%matplotlib inline
import random
from datetime import datetime
import matplotlib.dates as mdates
import json

import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import geopandas as gpd
import geoplot
from fbprophet import Prophet
slo = pd.read_excel("../input/last-data/stevilo_potrjenih_primerov_slo.xlsx")
gostota = pd.read_excel("..//input/cov-slo/gostota1.xlsx")
people_world = pd.read_excel("..//input/owidcoviddata1xlsx/owid-covid-data.xlsx")
cov= pd.read_csv("..//input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
sr= pd.read_excel("..//input/last-data/statistine_regije.xlsx")
cov19 = pd.read_csv("..//input/novel-corona-virus-2019-dataset/covid_19_data.csv")
age= pd.read_excel("..//input/last-data/age.xlsx")
slovsi = pd.read_excel("..//input/slo-vsi/COVID-19-vsi-podatki.xlsx")
srgeo= gpd.read_file('..//input/sr-geojson/SR.geojson')
new_header = slo.iloc[0] #grab the first row for the header
slo = slo[1:] #take the data less the header row
slo.columns = new_header
slo.columns=['Datum prijave', 'Dnevno število testiranj', 'Skupno število testiranj',
       'Dnevno število testiranj', 'Skupno število testiranj', 'Moški',
       'Ženske', 'Skupaj', 'Moški', 'Ženske', 'Skupaj ALL']
people_world= people_world.loc[:16813,:]
cov.isnull().sum()
sns.boxplot(x="gender", y = "age", data = cov)
cov["gender"].fillna("bla", inplace = True)
cov["gender"] = cov["gender"].map(lambda i: np.random.randint(2) if i =="bla" else i )
cov["gender"] = cov["gender"].replace({"male" : 0 , "female" : 1})
cov["age"].fillna("bla", inplace = True)
cov["age"] = cov["age"].map(lambda i: np.random.randint(45,56) if i ==  "bla" else i )
age_female_world = cov[cov["gender"]==1]["age"]
age_male_world = cov[cov["gender"]==0]["age"]
group1=[]
group2=[]
group3=[]
group4=[]
group5=[]
group6=[]
group7=[]
group8=[]
group9=[]
group10=[]
for i in age_male_world:
    if 0<i<4: 
        group1.append(i),
    elif 5<i<14: 
        group2.append(i),
    elif 15<i<24: 
        group3.append(i),
    elif 25<i<34:
        group4.append(i),
    elif 35<i<44: 
        group5.append(i),
    elif 45<i<54:
        group6.append(i),
    elif 55<i<64:
        group7.append(i),
    elif 65<i<74:
        group8.append(i),
    elif 75<i<84: 
        group9.append(i),
    else:
        group10.append(i)

group1= len(group1)
group2=len(group2)
group3=len(group3)
group4=len(group4)
group5=len(group5)
group6=len(group6)
group7=len(group7)
group8=len(group8)
group9=len(group9)
group10=len(group10)
print(group10,group9,group8,group7,group6,group5,group4,group3,group2,group1)
groupf1=[]
groupf2=[]
groupf3=[]
groupf4=[]
groupf5=[]
groupf6=[]
groupf7=[]
groupf8=[]
groupf9=[]
groupf10=[]
for i in age_female_world:
    if 0<i<4: 
        groupf1.append(i),
    elif 5<i<14: 
        groupf2.append(i),
    elif 15<i<24: 
        groupf3.append(i),
    elif 25<i<34:
        groupf4.append(i),
    elif 35<i<44: 
        groupf5.append(i),
    elif 45<i<54:
        groupf6.append(i),
    elif 55<i<64:
        groupf7.append(i),
    elif 65<i<74:
        groupf8.append(i),
    elif 75<i<84: 
        groupf9.append(i),
    else:
        groupf10.append(i)
groupf1= len(groupf1)
groupf2= len(groupf2)
groupf3=len(groupf3)
groupf4=len(groupf4)
groupf5=len(groupf5)
groupf6=len(groupf6)
groupf7=len(groupf7)
groupf8=len(groupf8)
groupf9=len(groupf9)
groupf10=len(groupf10)
print(groupf10,groupf9,groupf8,groupf7,groupf6,groupf5,groupf4,groupf3,groupf2,groupf1)
df = {'age group': ['85+','75-84',"65-74",'55-64','45-54','35-44','25-34','15-24','5-14','0-4'],
  'female': [162,11,31,54,31,38,36,15,2,2]} 
d = {'age group': ['85+','75-84',"65-74",'55-64','45-54','35-44','25-34','15-24','5-14','0-4'],
  'male': [251,12,35,49,51,57,48,9,5,3]}
world_male_age = pd.DataFrame(d)
world_female_age = pd.DataFrame(df)
fig, axarr  = plt.subplots(1,2,figsize = (18,5))
plt.subplots_adjust(wspace=0.08)
plt.suptitle("Number of infected people by age until 29.5.2020: Slovenia", size=16)
AgeClass = ['85+','75-84',"65-74",'55-64','45-54','35-44','25-34','15-24','5-14','0-4']

plotf = sns.barplot(x=age["Ženske"], y=age["Starostne skupine"],data = age, label = "Female",color = "r", alpha = .5,order=AgeClass,ax=axarr[1])
for i in plotf .patches:
    plotf .text(i.get_width()+1, i.get_y()+.55, s=format(int(i.get_width())),fontsize=12,color='black')
plotf.set_ylabel(" ")
plotf.set_xlabel("Female",fontsize=16)

plotm =  sns.barplot(x=age["Moški"], y="Starostne skupine",data = age,label = "Male", color = "b",order=AgeClass, alpha = .5, ax=axarr[0])
  

for i in plotm.patches:
    plotm.text(i.get_width()+5, i.get_y()+.55, s=format(int(i.get_width())),
             fontsize=12,color='black')
plotm.set_yticklabels([])   # Hide the left y-axis tick-labels
plotm.set_ylabel(" ")
plotm.set_xlabel("Male",fontsize=16)
plotm.invert_xaxis()   # labels read left to right


fig, axarr  = plt.subplots(1,2,figsize = (18,5))
plt.subplots_adjust(wspace=0.08)
plt.suptitle("Number of infected people by age: World*", size=16)
AgeClass = ['85+','75-84',"65-74",'55-64','45-54','35-44','25-34','15-24','5-14','0-4']

plotf = sns.barplot(x="female", y="age group",data = world_female_age, label = "female",color = "r", alpha = .5,ax=axarr[1])
for i in plotf .patches:
    plotf .text(i.get_width()+1, i.get_y()+.55, s=format(int(i.get_width())),fontsize=12,color='black')
plotf.set_ylabel(" ")
plotf.set_xlabel("Female",fontsize=16)

plotm =  sns.barplot(x="male", y="age group",data = world_male_age,label = "male", color = "b", alpha = .5, ax=axarr[0])
  

for i in plotm.patches:
    plotm.text(i.get_width()+12, i.get_y()+.55, s=format(int(i.get_width())),
             fontsize=12,color='black')
plotm.set_yticklabels([])   # Hide the left y-axis tick-labels
plotm.set_ylabel(" ")
plotm.set_xlabel("Male",fontsize=16)
plotm.invert_xaxis()   # labels read left to right
slovsi1 = slovsi.copy()
slovsi1["Datum"] = slovsi1["Datum"].dt.strftime("%d-%b")

plt.figure(figsize= (14,10))
positiveslo = sns.barplot(x= "Datum", y = "Dnevno število pozitivnih oseb", data = slovsi1, color =  "Green", alpha = .5)
plt.xticks(rotation=90)
for p in positiveslo.patches:
    positiveslo.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.title("New cases per day: Slovenia")
total_cases_world = pd.DataFrame( people_world.groupby("date")["new_cases"].sum().reset_index())
total_cases_world["date"] = pd.to_datetime(total_cases_world["date"])
total_cases_world["date"] = total_cases_world["date"].dt.strftime("%d-%b")
plt.figure(figsize= (18,14))
positive_world = sns.barplot(x=total_cases_world.iloc[24:,0], y = "new_cases", data = total_cases_world, color =  "Green", alpha = .5)
plt.xticks(rotation=90, fontsize=8)
positive_world .set_ylabel("Positive (daily)")
positive_world .set_xlabel("Date")
plt.title("New cases per day: World",fontsize = 18)
plt.figure(figsize= (14,10))
hospitalized  = sns.barplot(x= "Datum", y = "Skupno število hospitaliziranih oseb na posamezni dan",label = "All hospitalized", data = slovsi1, color =  "r", alpha = .5)
for p in hospitalized.patches:
    hospitalized.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 11),fontsize = 9, textcoords = 'offset points')
intensive = sns.barplot(x= "Datum", y = "Skupno število oseb na intenzivni negi na posamezni dan",label = "All persons in intensive care", data = slovsi1, color =  "Green", alpha = .5)
for p in intensive.patches:
    intensive.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 11),fontsize = 9, textcoords = 'offset points')
plt.title("Hospitalized/intensive care : Slovenia", fontsize = 18)
plt.xticks(rotation=90)
plt.legend()
plt.figure(figsize= (14,10))
tested_daily= sns.barplot(x= "Datum", y = "Dnevno število testiranj",label = "Tested (daily)", data = slovsi1, color =  "r", alpha = .5)
for p in tested_daily.patches:
    tested_daily.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10),fontsize = 8, textcoords = 'offset points')
positive_daily= sns.barplot(x= "Datum", y = "Dnevno število pozitivnih oseb",label = "Positive (daily)",data = slovsi1, color =  "Green", alpha = .5)
for p in positive_daily.patches:
    positive_daily.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 8,textcoords = 'offset points')

plt.title("Daily tested/positive : Slovenia", fontsize = 18)    
plt.xticks(rotation=90)
plt.legend()
plt.figure(figsize=(18,12))
sns.set_style("darkgrid")
positive_slo = sns.lineplot(x= "Datum",y= "Skupno število pozitivnih oseb",data=slovsi, color = "r",label = " Total positive")
deaths_slo = sns.lineplot(x= "Datum",y= "Skupno število umrlih",data=slovsi, color = "black",label = "Total deaths")


plt.ylabel("Total cases")
plt.xlabel("Date")
plt.legend()
plt.title("Slovenia", fontsize = 16)
plt.xticks(rotation=45,horizontalalignment='right',fontsize = 9)
deaths_slo.set(xticks=slovsi.Datum.values)   
deaths_slo.xaxis.set_major_formatter(mdates.DateFormatter('%d.%b'))
total_deaths_world = pd.DataFrame(people_world.groupby(["date"])["total_deaths"].sum().reset_index())
total_deaths_world["date"] = pd.to_datetime(total_deaths_world["date"])
total_cases_world1 = pd.DataFrame( people_world.groupby("date")["total_cases"].sum().reset_index())
total_cases_world1["date"] = pd.to_datetime(total_cases_world1["date"])
plt.figure(figsize=(18,14))
positive_world= sns.lineplot(x= "date",y= "total_cases",data=total_cases_world1, color = "r", label = "Total positive")
deaths_world = sns.lineplot(x= "date",y= "total_deaths",data=total_deaths_world, color = "black", label = "Total deaths")


plt.xticks(rotation=45,horizontalalignment='right')
positive_world.set_ylabel("Total cases in milions")
positive_world.set_xlabel("Date")
plt.legend()
plt.title("World", fontsize = 18)
positive_world.set(xticks=total_deaths_world.date.values)
positive_world.xaxis.set_major_formatter(mdates.DateFormatter('%d.%b')) 
positive_world.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.figure(figsize=(18,10))
tested_slo = sns.lineplot(x= "Datum",y= "Skupno število testiranj",data=slovsi, color = "b",label = "Tested")
plt.xticks(rotation=45,horizontalalignment='right', fontsize =9)
plt.title("Slovenia", fontsize = 16)
plt.legend(loc=2)
tested_slo.set(xticks=slovsi.Datum.values)
tested_slo.xaxis.set_major_formatter(mdates.DateFormatter('%d.%b')) 
total_tests_world = pd.DataFrame( people_world.groupby("date")["total_tests"].sum().reset_index())
total_tests_world["date"] = pd.to_datetime(total_tests_world["date"])
plt.figure(figsize=(14,8))
tested_world = sns.lineplot(x=total_tests_world["date"][:-2],y = "total_tests", data = total_tests_world,color = "b",label = "Tested")
plt.xticks(rotation=45, horizontalalignment='right')
plt.title("World", fontsize=16)
tested_world.set_ylabel("Tested(All) in milions")
tested_world.set_xlabel("Date")
tested_world.set(xticks= total_tests_world.date.values)
tested_world.xaxis.set_major_formatter(mdates.DateFormatter('%d.%b')) 
tested_world.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.figure(figsize=(12,10))
g = sns.barplot(x="št. Do 28.4", y="name",data = sr.sort_values("št. Do 28.4", ascending = False),color = "g", alpha =.5)
g.set_ylabel("The name of the statistical region")
g.set_xlabel("Number of infected")
for i in g.patches:
    g.text(i.get_width()+6, i.get_y()+.55, s=format(round(i.get_width())),
             fontsize=12,color='black')
g.set_title ("Infected by statistical regions until 29.5.2020", fontsize =12)

srgeo['coords'] = srgeo['geometry'].apply(lambda x: x.representative_point().coords[:])
srgeo['coords'] = [coords[0] for coords in srgeo['coords']]
sr = pd.concat([sr, srgeo['coords'] ], axis = 1)

sr_plot= geoplot.choropleth(srgeo,figsize=(16,8),hue = sr["št. Do 28.4"],  cmap='Pastel2', legend=True)
plt.title("Infected by statistical regions until 29.5.2020", fontsize=12)
for idx, row in sr[:12].iterrows():
        plt.annotate(s=row['št. Do 28.4'],xy=row['coords'],horizontalalignment='center', fontsize=16, color = "black" )

population_density= pd.concat([gostota, srgeo['coords'] ], axis = 1)
population_density_plot = geoplot.choropleth(srgeo,figsize=(16,8),hue = gostota["dnsty_p"],  cmap='Pastel2', legend=True)
plt.title("Population density of Slovenia per km²")
for idx, row in population_density.iterrows():
        plt.annotate(s=row["dnsty_p"],xy=row['coords'],horizontalalignment='center', fontsize=16, color = "black" )
world = cov19.groupby("Country/Region")[["Confirmed","Deaths","Recovered"]].max().reset_index()
init_notebook_mode(connected=True) 
data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
      
        reversescale = True,
        locations = world['Country/Region'],
        locationmode = "country names",
        z = world['Confirmed'],
        text = world['Country/Region'],
        colorbar = {'title' : "Confirmed"},
      ) 

layout = dict(title = "Total infected until 13.5.2020",
                geo = dict(showframe = False,projection = {'type':'mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
case_per_milion = pd.DataFrame(people_world.groupby("location")["total_cases_per_million"].max().reset_index())
data = dict(
        type = 'choropleth',
        colorscale ="Reds",
        zmax =300,
        zmin = 0,
        zsrc ="europe",
       
        locations = people_world['location'],
        locationmode = "country names",
        z = people_world['population_density'],
        marker = go.choropleth.Marker(line = go.choropleth.marker.Line(color = 'rgb(180,180,180)', width = 0.5) ),
        colorbar = {'title' : "population density per km²"},
      ) 

layout = dict(title = "Population density per km²",
                geo = dict(showframe = False,projection = {'type':'mercator'})
             )

choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(
        type = 'choropleth',
        colorscale ="Rainbow",
        zmax =5000,
        zmin = 0,
        zsrc ="europe",
       
        locations = case_per_milion['location'],
        locationmode = "country names",
        z = case_per_milion["total_cases_per_million"],
        marker = go.choropleth.Marker(line = go.choropleth.marker.Line(color = 'rgb(180,180,180)', width = 0.5) ),
        colorbar = {'title' : "Total infected per million"},
      ) 

layout = dict(title = "Total infected cases per million population",
                geo = dict(showframe = False,projection = {'type':'mercator'})
             )

choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
slovsi = slovsi.copy()
slovsi = slovsi.rename(columns = {"Datum":"ds","Skupno število pozitivnih oseb":"y"}) 
pred = slovsi[["ds","y"]]
pred.tail()
model = Prophet(interval_width=0.95)
model_positive = model.fit(pred)
future = model_positive.make_future_dataframe(periods=14)
future.tail()
forecast = model_positive.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model.plot(forecast, xlabel='Date', ylabel='Value')
ax = fig.gca()
ax.set_title("Forecast of infected in Slovenia", size=20)


