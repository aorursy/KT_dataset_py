import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.integrate import odeint

from plotly.offline import iplot, init_notebook_mode
import math
import bokeh 
import matplotlib.pyplot as plt
import plotly.express as px
from urllib.request import urlopen
import json
from dateutil import parser
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row, column
from bokeh.resources import INLINE
from bokeh.io import output_notebook
from bokeh.models import Span
import warnings
warnings.filterwarnings("ignore")
output_notebook(resources=INLINE)
covid=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid.head()
#Extracting India's data 
covid_india=covid[covid['Country/Region']=="India"]

#Extracting other countries for comparison of worst affected countries
covid_spain=covid[covid['Country/Region']=="Spain"]
covid_us=covid[covid['Country/Region']=="US"]
covid_italy=covid[covid['Country/Region']=="Italy"]
covid_iran=covid[covid['Country/Region']=="Iran"]
covid_france=covid[covid['Country/Region']=="France"]
covid_uk=covid[covid['Country/Region']=="UK"]

#Extracting data of neighbouring countries
covid_pak=covid[covid['Country/Region']=="Pakistan"]
covid_china=covid[covid['Country/Region']=="Mainland China"]
covid_afg=covid[covid['Country/Region']=="Afghanistan"]
covid_nepal=covid[covid['Country/Region']=="Nepal"]
covid_bhutan=covid[covid['Country/Region']=="Bhutan"]
covid_lanka=covid[covid["Country/Region"]=="Sri Lanka"]
covid_ban=covid[covid["Country/Region"]=="Bangladesh"]
#Converting the date into Datetime format
covid_india["ObservationDate"]=pd.to_datetime(covid_india["ObservationDate"])
covid_spain["ObservationDate"]=pd.to_datetime(covid_spain["ObservationDate"])
covid_us["ObservationDate"]=pd.to_datetime(covid_us["ObservationDate"])
covid_italy["ObservationDate"]=pd.to_datetime(covid_italy["ObservationDate"])
covid_iran["ObservationDate"]=pd.to_datetime(covid_iran["ObservationDate"])
covid_france["ObservationDate"]=pd.to_datetime(covid_france["ObservationDate"])
covid_uk["ObservationDate"]=pd.to_datetime(covid_uk["ObservationDate"])

covid_pak["ObservationDate"]=pd.to_datetime(covid_pak["ObservationDate"])
covid_china["ObservationDate"]=pd.to_datetime(covid_china["ObservationDate"])
covid_afg["ObservationDate"]=pd.to_datetime(covid_afg["ObservationDate"])
covid_nepal["ObservationDate"]=pd.to_datetime(covid_nepal["ObservationDate"])
covid_bhutan["ObservationDate"]=pd.to_datetime(covid_bhutan["ObservationDate"])
covid_lanka["ObservationDate"]=pd.to_datetime(covid_lanka["ObservationDate"])
covid_ban["ObservationDate"]=pd.to_datetime(covid_ban["ObservationDate"])
#Grouping the data based on the Date 
india_datewise=covid_india.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
spain_datewise=covid_spain.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
us_datewise=covid_us.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
italy_datewise=covid_italy.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
iran_datewise=covid_iran.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
france_datewise=covid_france.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
uk_datewise=covid_uk.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

pak_datewise=covid_pak.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
china_datewise=covid_china.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
afg_datewise=covid_afg.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
nepal_datewise=covid_nepal.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
bhutan_datewise=covid_bhutan.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
lanka_datewise=covid_lanka.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
ban_datewise=covid_ban.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
#Adding week column to perfom weekly analysis further ahead
india_datewise["WeekofYear"]=india_datewise.index.weekofyear
spain_datewise["WeekofYear"]=spain_datewise.index.weekofyear
us_datewise["WeekofYear"]=us_datewise.index.weekofyear
italy_datewise["WeekofYear"]=italy_datewise.index.weekofyear
iran_datewise["WeekofYear"]=iran_datewise.index.weekofyear
france_datewise["WeekofYear"]=france_datewise.index.weekofyear
uk_datewise["WeekofYear"]=uk_datewise.index.weekofyear

pak_datewise["WeekofYear"]=pak_datewise.index.weekofyear
china_datewise["WeekofYear"]=china_datewise.index.weekofyear
afg_datewise["WeekofYear"]=afg_datewise.index.weekofyear
nepal_datewise["WeekofYear"]=nepal_datewise.index.weekofyear
bhutan_datewise["WeekofYear"]=bhutan_datewise.index.weekofyear
lanka_datewise["WeekofYear"]=lanka_datewise.index.weekofyear
ban_datewise["WeekofYear"]=ban_datewise.index.weekofyear
india_datewise["Days Since"]=(india_datewise.index-india_datewise.index[0])
india_datewise["Days Since"]=india_datewise["Days Since"].dt.days
grouped_country=covid.groupby(["Country/Region","ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
grouped_country["Active Cases"]=grouped_country["Confirmed"]-grouped_country["Recovered"]-grouped_country["Deaths"]
grouped_country["log_confirmed"]=np.log(grouped_country["Confirmed"])
grouped_country["log_active"]=np.log(grouped_country["Active Cases"])
print("Number of Confirmed Cases",india_datewise["Confirmed"].iloc[-1])
print("Number of Recovered Cases",india_datewise["Recovered"].iloc[-1])
print("Number of Death Cases",india_datewise["Deaths"].iloc[-1])
print("Number of Active Cases",india_datewise["Confirmed"].iloc[-1]-india_datewise["Recovered"].iloc[-1]-india_datewise["Deaths"].iloc[-1])
print("Number of Closed Cases",india_datewise["Recovered"].iloc[-1]+india_datewise["Deaths"].iloc[-1])
print("Approximate Number of Confirmed Cases per day",round(india_datewise["Confirmed"].iloc[-1]/india_datewise.shape[0]))
print("Approximate Number of Recovered Cases per day",round(india_datewise["Recovered"].iloc[-1]/india_datewise.shape[0]))
print("Approximate Number of Death Cases per day",round(india_datewise["Deaths"].iloc[-1]/india_datewise.shape[0]))
print("Number of New Cofirmed Cases in last 24 hours are",india_datewise["Confirmed"].iloc[-1]-india_datewise["Confirmed"].iloc[-2])
print("Number of New Recoverd Cases in last 24 hours are",india_datewise["Recovered"].iloc[-1]-india_datewise["Recovered"].iloc[-2])
print("Number of New Death Cases in last 24 hours are",india_datewise["Deaths"].iloc[-1]-india_datewise["Deaths"].iloc[-2])
plt.figure(figsize=(15,5))
sns.barplot(x=india_datewise.index.date,y=india_datewise["Confirmed"]-india_datewise["Recovered"]-india_datewise["Deaths"])
plt.xticks(rotation=90)
plt.ylabel("Number of Cases")
plt.xlabel("Date")
plt.title("Distribution of Number of Active Cases in India over Date")
plt.figure(figsize=(15,5))
sns.barplot(x=india_datewise.index.date,y=india_datewise["Recovered"]+india_datewise["Deaths"])
plt.xticks(rotation=90)
plt.ylabel("Number of Cases")
plt.xlabel("Date")
plt.title("Distribution of Number of Closed Cases in India over Date")
plt.figure(figsize=(10,5))
plt.plot(india_datewise["Confirmed"],label="Confirmed Cases",marker='o')
plt.plot(india_datewise["Recovered"],label="Recovered Cases",marker='*')
plt.plot(india_datewise["Deaths"],label="Death Cases",marker="^")
plt.xticks(rotation=90)
plt.ylabel("Number of Cases")
plt.xlabel("Date")
plt.title("Growth of different types of cases in India")
plt.legend()
fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(15,5))
ax1.plot((india_datewise["Recovered"]/india_datewise["Confirmed"])*100,label="Recovery Rate",linewidth=3)
ax1.axhline(((india_datewise["Recovered"]/india_datewise["Confirmed"])*100).mean(),linestyle='--',color='black',label="Mean Recovery Rate")
ax1.set_xlabel("Date")
ax1.set_ylabel("Recovery Rate")
ax1.set_title("Recovery Rate of India over Date")
ax1.legend()
ax2.plot((india_datewise["Deaths"]/india_datewise["Confirmed"])*100,label="Mortality Rate",linewidth=3)
ax2.axhline(((india_datewise["Deaths"]/india_datewise["Confirmed"])*100).mean(),linestyle='--',color='black',label="Mean Mortality Rate")
ax2.set_xlabel("Date")
ax2.set_ylabel("Mortality Rate")
ax2.set_title("Mortality Rate of India over Date")
ax2.legend()
plt.figure(figsize=(14,6))
plt.plot(india_datewise["Confirmed"]/india_datewise["Confirmed"].shift(),linewidth=3,label="Growth Factor of Confirmed Cases")
plt.plot(india_datewise["Recovered"]/india_datewise["Recovered"].shift(),linewidth=3,label="Growth Factor of Recovered Cases")
plt.plot(india_datewise["Deaths"]/india_datewise["Deaths"].shift(),linewidth=3,label="Growth Factor of Death Cases")
plt.axhline(1,linestyle='--',color='black',label="Baseline")
plt.legend()
plt.title("Datewise Growth Factor of different types of Cases in India")
plt.xticks(rotation=90)
plt.figure(figsize=(12,6))
plt.plot(india_datewise["Confirmed"].diff().fillna(0),linewidth=3,label="Confirmed Cases")
plt.plot(india_datewise["Recovered"].diff().fillna(0),linewidth=3,label="Recovered Cases")
plt.plot(india_datewise["Deaths"].diff().fillna(0),linewidth=3,label="Death Cases")
plt.ylabel("Increase in Number of Cases")
plt.xlabel("Date")
plt.title("Daily increase in different types of cases in India")
plt.xticks(rotation=90)
plt.legend()
covid_India_cases = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
covid_India_cases.rename(columns={'State/UnionTerritory': 'State', 'Cured': 'Recovered', 'Confirmed': 'Confirmed'}, inplace=True)

statewise_cases = pd.DataFrame(covid_India_cases.groupby(['State'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index())
statewise_cases["Country"] = "India" # in order to have a single root node
fig = px.treemap(statewise_cases, path=['Country','State'], values='Confirmed',
                  color='Confirmed', hover_data=['State'],
                  color_continuous_scale='Rainbow')
fig.show()
import IPython
IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1977187" data-url="https://flo.uri.sh/visualisation/1977187/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
covid_India_cases = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
covid_India_cases.rename(columns={'State/UnionTerritory': 'State', 'Cured': 'Recovered', 'Confirmed': 'Confirmed'}, inplace=True)

statewise_cases = pd.DataFrame(covid_India_cases.groupby(['State'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index())
last=statewise_cases
pos=pd.read_csv('../input/utm-of-india/UTM ZONES of INDIA.csv')
ind_grp=last.merge(pos , left_on='State', right_on='State / Union Territory')
import folium
map = folium.Map(location=[20.5937, 78.9629], zoom_start=4,tiles='cartodbpositron')

for lat, lon,state,Confirmed,Recovered,Deaths in zip(ind_grp['Latitude'], ind_grp['Longitude'],ind_grp['State'],ind_grp['Confirmed'],ind_grp['Recovered'],ind_grp['Deaths']):
    folium.CircleMarker([lat, lon],
                        radius=5,
                        color='YlOrRd',
                      popup =(
                    'State: ' + str(state) + '<br>'
                    'Confirmed: ' + str(Confirmed) + '<br>'
                      'Recovered: ' + str(Recovered) + '<br>'
                      'Deaths: ' + str(Deaths) + '<br>'),

                        fill_color='red',
                        fill_opacity=0.7 ).add_to(map)
map
from folium.plugins import HeatMap
m = folium.Map(location = [20.5937, 78.9629], zoom_start = 4,tiles='cartodbpositron',columns = ['State/UnionTerritory','Confirmed'],)

heat_data = [[row['Latitude'],row['Longitude']] for index, row in ind_grp.iterrows()]
HeatMap(heat_data,radius=16.5, blur = 5.5).add_to(m)

m
ind_map=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
#ind_map.head()
pos=pd.read_csv('../input/utm-of-india/UTM ZONES of INDIA.csv')
ind_map1=ind_map.merge(pos , left_on='State/UnionTerritory', right_on='State / Union Territory')
#ind_map1.head()
#ind_map = ind_grp
ind_map1  = ind_map1.groupby(['Date', 'State/UnionTerritory','Latitude','Longitude'])['Confirmed'].sum()


ind_map1 = ind_map1.reset_index()
ind_map1.head()
ind_map1['size'] = ind_map1['Confirmed']*90000000
ind_map1
fig = px.scatter_mapbox(ind_map1, lat="Latitude", lon="Longitude",
                     color="Confirmed", size='size',hover_data=['State/UnionTerritory'],
                     color_continuous_scale='burgyl', animation_frame="Date", 
                     title='Spread total cases over time in India')
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=3, mapbox_center = {"lat":20.5937,"lon":78.9629})
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()
import pandas as pd
import numpy as np
import datetime
import requests
import warnings

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import squarify
import plotly_express as px


from IPython.display import Image
warnings.filterwarnings('ignore')
%matplotlib inline
age_details = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')
india_covid_19 = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
hospital_beds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
#ICMR_details = pd.read_csv('../input/covid19-in-india/ICMRTestingDetails.csv')
ICMR_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')
state_testing = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')

india_covid_19['Date'] = pd.to_datetime(india_covid_19['Date'])
state_testing['Date'] = pd.to_datetime(state_testing['Date'])
cumulative_df = india_covid_19.groupby(["State/UnionTerritory", "Date"])["Confirmed", "Deaths", "Cured"].sum().reset_index()
cumulative_df["Date"] = pd.to_datetime(cumulative_df["Date"] , format="%m/%d/%Y").dt.date
cumulative_df = cumulative_df.sort_values(by="Date").reset_index(drop=True)
start_date = datetime.date(2020, 3, 10)
cumulative_df = cumulative_df[cumulative_df["Date"]>=start_date]
cumulative_df["Date"] = cumulative_df["Date"].astype(str)

fig = px.scatter(cumulative_df, x="Confirmed", y="Deaths", animation_frame="Date", animation_group="State/UnionTerritory",
           size="Confirmed", color="State/UnionTerritory", hover_name="State/UnionTerritory",
           log_x=False, size_max=55, range_x=[0,15000], range_y=[-20,800])

layout = go.Layout(
    title=go.layout.Title(
        text="Changes in number of confirmed & death cases over time in India states",
        x=0.5
    ),
    font=dict(size=14),
    xaxis_title = "Total number of confirmed cases",
    yaxis_title = "Total number of death cases"
)

fig.update_layout(layout)

fig.show()
labels = ['Missing', 'Male', 'Female']
sizes = []
sizes.append(individual_details['gender'].isnull().sum())
sizes.append(list(individual_details['gender'].value_counts())[0])
sizes.append(list(individual_details['gender'].value_counts())[1])

explode = (0, 0.1, 0)
colors = ['#ffcc99','#66b3ff','#ff9999']

plt.figure(figsize= (15,10))
plt.title('Percentage of Gender',fontsize = 20)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)
plt.axis('equal')
plt.tight_layout()


labels = ['Male', 'Female']
sizes = []
sizes.append(list(individual_details['gender'].value_counts())[0])
sizes.append(list(individual_details['gender'].value_counts())[1])

explode = (0.1, 0)
colors = ['#66b3ff','#ff9999']

plt.figure(figsize= (10,8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)

plt.title('Percentage of Gender (Ignoring the Missing Values)',fontsize = 20)
plt.axis('equal')
plt.tight_layout()


df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
data = df.copy()
data['Date'] = data['Date'].apply(pd.to_datetime)
data.drop(['Sno', 'Time'],axis=1,inplace=True)

# collect present data
from datetime import date
data_apr = data[data['Date'] > pd.Timestamp(date(2020,4,12))]

# prepaing data state wise
state_cases = data_apr.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()
state_cases['Active'] = state_cases['Confirmed'] - (state_cases['Deaths']- state_cases['Cured'])
state_cases["Death Rate (per 100)"] = np.round(100*state_cases["Deaths"]/state_cases["Confirmed"],2)
state_cases["Cure Rate (per 100)"] = np.round(100*state_cases["Cured"]/state_cases["Confirmed"],2)
state_details = pd.pivot_table(df, values=['Confirmed','Deaths','Cured'], index='State/UnionTerritory', aggfunc='max')
state_details['Recovery Rate'] = round(state_details['Cured'] / state_details['Confirmed'],2)
state_details['Deaths']['Madhya Pradesh#']=119

state_details['Deaths']=state_details['Deaths'].astype(np.float32)
#state_details=state_details.reset_index()
state_details['Active']=state_details['Confirmed']-state_details['Cured']-state_details['Deaths']

state_details['Death Rate'] = round(state_details['Deaths'] /state_details['Confirmed'], 2)
state_details = state_details.sort_values(by='Confirmed', ascending= False)
#state_details.style.background_gradient(cmap='PuBuGn')
state_details.style.bar(subset=['Confirmed'], color='#FDD017')\
                    .bar(subset=['Cured'], color='lime')\
                    .bar(subset=['Deaths'], color='red')\
                    .bar(subset=['Active'], color='#0000FF')\
                    .bar(subset=['Recovery Rate'], color='#B1FB17')\
                    .bar(subset=['Death Rate'], color='#C0C0C0')
state_testing = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')
state_testing
labs = pd.read_csv("../input/covid19-in-india/ICMRTestingLabs.csv")
fig = px.treemap(labs, path=['state','city'],
                  color='city', hover_data=['lab','address'],
                  color_continuous_scale='reds')
fig.show()
testing=state_testing.groupby('State').sum().reset_index()
testing=testing.sort_values(['TotalSamples'], ascending=True)
fig = px.bar(testing, 
             x="TotalSamples",
             y="State", 
             orientation='h',
             height=800,
             title='Testing statewise insight')
fig.show()
import plotly.express as px
values = list(ICMR_labs['state'].value_counts())
names = list(ICMR_labs['state'].value_counts().index)
df = pd.DataFrame(list(zip(values, names)), 
               columns =['values', 'names'])

fig = px.bar(df, 
             x="values", 
             y="names", 
             orientation='h',
             height=1000,
             title='ICMR Testing Centers in each State')
fig.show()

plt.figure(figsize=(20,60))
plt.subplot(4,1,1)
hospital_beds=hospital_beds.sort_values('NumUrbanHospitals_NHP18', ascending= False)
sns.barplot(data=hospital_beds,y='State/UT',x='NumUrbanHospitals_NHP18',color=sns.color_palette('RdBu')[0])
plt.title('Urban Hospitals per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospital_beds.shape[0]):
    count = hospital_beds.iloc[i]['NumUrbanHospitals_NHP18']
    plt.text(count+10,i,count,ha='center',va='center')

plt.subplot(4,1,2)
hospital_beds=hospital_beds.sort_values('NumRuralHospitals_NHP18', ascending= False)
sns.barplot(data=hospital_beds,y='State/UT',x='NumRuralHospitals_NHP18',color=sns.color_palette('RdBu')[1])
plt.title('Rural Hospitals per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospital_beds.shape[0]):
    count = hospital_beds.iloc[i]['NumRuralHospitals_NHP18']
    plt.text(count+100,i,count,ha='center',va='center')

plt.subplot(4,1,3)
hospitalBeds=hospital_beds.sort_values('NumUrbanBeds_NHP18', ascending= False)
sns.barplot(data=hospitalBeds,y='State/UT',x='NumUrbanBeds_NHP18',color=sns.color_palette('RdBu')[5])
plt.title('Urban Beds per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospitalBeds.shape[0]):
    count = hospitalBeds.iloc[i]['NumUrbanBeds_NHP18']
    plt.text(count+1500,i,count,ha='center',va='center')

plt.subplot(4,1,4)
hospitalBeds=hospitalBeds.sort_values('NumRuralBeds_NHP18', ascending= False)
sns.barplot(data=hospitalBeds,y='State/UT',x='NumRuralBeds_NHP18',color=sns.color_palette('RdBu')[3])
plt.title('Rural Beds per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospitalBeds.shape[0]):
    count = hospitalBeds.iloc[i]['NumRuralBeds_NHP18']
    plt.text(count+1500,i,count,ha='center',va='center')

plt.show()
plt.tight_layout()

covid_India_cases = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
covid_India_cases=covid_India_cases.dropna()
covid_India_cases.rename(columns={'State/UnionTerritory': 'State', 'Cured': 'Recovered', 'Confirmed': 'Confirmed'}, inplace=True)

covid_India_cases = covid_India_cases.fillna('unknow')
top_country = covid_India_cases.loc[covid_India_cases['Date'] == covid_India_cases['Date'].iloc[-1]]
top_country = top_country.groupby(['State'])['Confirmed'].sum().reset_index()
top_country = top_country.sort_values('Confirmed', ascending=False)
top_country = top_country[:30]
top_country_codes = top_country['State']
top_country_codes = list(top_country_codes)

#countries = virus_data.loc[virus_data['Country'] in top_country_codes]
countries = covid_India_cases[covid_India_cases['State'].isin(top_country_codes)]
countries_day = countries.groupby(['Date','State'])['Confirmed','Deaths','Recovered'].sum().reset_index()


exponential_line_x = []
exponential_line_y = []
for i in range(16):
    exponential_line_x.append(i)
    exponential_line_y.append(i)
    
################################    Maharashtra    #################
Maharashtra = countries_day.loc[countries_day['State']=='Maharashtra']
Maharashtra=Maharashtra.sort_values('Confirmed',ascending=True)

new_confirmed_cases_Maharashtra = []
new_confirmed_cases_Maharashtra.append( list(Maharashtra['Confirmed'])[0] - list(Maharashtra['Deaths'])[0] 
                           - list(Maharashtra['Recovered'])[0] )

for i in range(1,len(Maharashtra)):

    new_confirmed_cases_Maharashtra.append( list(Maharashtra['Confirmed'])[i] - 
                                     list(Maharashtra['Deaths'])[i] - 
                                     list(Maharashtra['Recovered'])[i])
    
#######################   Gujarat   ############
Gujarat = countries_day.loc[countries_day['State']=='Gujarat']
Gujarat=Gujarat.sort_values('Confirmed',ascending=True)

new_confirmed_cases_Gujarat = []
new_confirmed_cases_Gujarat.append( list(Gujarat['Confirmed'])[0] - list(Gujarat['Deaths'])[0] 
                           - list(Gujarat['Recovered'])[0] )

for i in range(1,len(Gujarat)):
    
    new_confirmed_cases_Gujarat.append( list(Gujarat['Confirmed'])[i] - 
                                  list(Gujarat['Deaths'])[i] - 
                                  list(Gujarat['Recovered'])[i])
    
###########################    Delhi   ##################3
Delhi = countries_day.loc[countries_day['State']=='Delhi']
Delhi=Delhi.sort_values('Confirmed',ascending=True)

new_confirmed_cases_Delhi = []
new_confirmed_cases_Delhi.append( list(Delhi['Confirmed'])[0] - list(Delhi['Deaths'])[0] 
                           - list(Delhi['Recovered'])[0] )

for i in range(1,len(Delhi)):
    
    new_confirmed_cases_Delhi.append( list(Delhi['Confirmed'])[i] - 
                                     list(Delhi['Deaths'])[i] - 
                                    list(Delhi['Recovered'])[i])
    
#############################    Madhya Pradesh    ################3
Madhya_Pradesh = countries_day.loc[countries_day['State']=='Madhya Pradesh']
Madhya_Pradesh=Madhya_Pradesh.sort_values('Confirmed',ascending=True)

new_confirmed_cases_Madhya_Pradesh = []
new_confirmed_cases_Madhya_Pradesh.append( list(Madhya_Pradesh['Confirmed'])[0] - list(Madhya_Pradesh['Deaths'])[0] 
                           - list(Madhya_Pradesh['Recovered'])[0] )

for i in range(1,len(Madhya_Pradesh)):
    
    new_confirmed_cases_Madhya_Pradesh.append( list(Madhya_Pradesh['Confirmed'])[i] - 
                                     list(Madhya_Pradesh['Deaths'])[i] - 
                                    list(Madhya_Pradesh['Recovered'])[i])
    
################################   Rajasthan  ##########
Rajasthan = countries_day.loc[countries_day['State']=='Rajasthan']
Rajasthan=Rajasthan.sort_values('Confirmed',ascending=True)

new_confirmed_cases_Rajasthan = []
new_confirmed_cases_Rajasthan.append( list(Rajasthan['Confirmed'])[0] - list(Rajasthan['Deaths'])[0] 
                           - list(Rajasthan['Recovered'])[0] )

for i in range(1,len(Rajasthan)):
    
    new_confirmed_cases_Rajasthan.append( list(Rajasthan['Confirmed'])[i] - 
                                     list(Rajasthan['Deaths'])[i] - 
                                    list(Rajasthan['Recovered'])[i])
    
#################################    Uttar Pradesh   ##############
Uttar_Pradesh = countries_day.loc[countries_day['State']=='Uttar Pradesh']
Uttar_Pradesh=Uttar_Pradesh.sort_values('Confirmed',ascending=True)

new_confirmed_cases_Uttar_Pradesh = []
new_confirmed_cases_Uttar_Pradesh.append( list(Uttar_Pradesh['Confirmed'])[0] - list(Uttar_Pradesh['Deaths'])[0] 
                           - list(Uttar_Pradesh['Recovered'])[0] )

for i in range(1,len(Uttar_Pradesh)):
    
    new_confirmed_cases_Uttar_Pradesh.append( list(Uttar_Pradesh['Confirmed'])[i] - 
                                     list(Uttar_Pradesh['Deaths'])[i] - 
                                    list(Uttar_Pradesh['Recovered'])[i])
    
#####################################  Tamil Nadu  ############
Tamil_Nadu = countries_day.loc[countries_day['State']=='Tamil Nadu']
Tamil_Nadu=Tamil_Nadu.sort_values('Confirmed',ascending=True)

new_confirmed_cases_Tamil_Nadu = []
new_confirmed_cases_Tamil_Nadu.append( list(Tamil_Nadu['Confirmed'])[0] - list(Tamil_Nadu['Deaths'])[0] 
                           - list(Tamil_Nadu['Recovered'])[0] )

for i in range(1,len(Tamil_Nadu)):
    
    new_confirmed_cases_Tamil_Nadu.append( list(Tamil_Nadu['Confirmed'])[i] - 
                                     list(Tamil_Nadu['Deaths'])[i] - 
                                    list(Tamil_Nadu['Recovered'])[i])
######################################## Andhra Pradesh ##############
Andhra_Pradesh = countries_day.loc[countries_day['State']=='Andhra Pradesh']
Andhra_Pradesh=Andhra_Pradesh.sort_values('Confirmed',ascending=True)

new_confirmed_cases_Andhra_Pradesh = []
new_confirmed_cases_Andhra_Pradesh.append( list(Andhra_Pradesh['Confirmed'])[0] - list(Andhra_Pradesh['Deaths'])[0] 
                           - list(Andhra_Pradesh['Recovered'])[0] )

for i in range(1,len(Andhra_Pradesh)):
    
    new_confirmed_cases_Andhra_Pradesh.append( list(Andhra_Pradesh['Confirmed'])[i] - 
                                     list(Andhra_Pradesh['Deaths'])[i] - 
                                    list(Andhra_Pradesh['Recovered'])[i])

########################################Telengana#####################

#Telengana = countries_day.loc[countries_day['State']=='Telengana']
#Telengana=Telengana.sort_values('Confirmed',ascending=True)

#new_confirmed_cases_Telengana = []
#new_confirmed_cases_Telengana.append( list(Telengana['Confirmed'])[0] - list(Telengana['Deaths'])[0] 
#                           - list(Telengana['Recovered'])[0] )

#for i in range(1,len(Telengana)):
    
#    new_confirmed_cases_Telengana.append( list(Telengana['Confirmed'])[i] - 
#                                     list(Telengana['Deaths'])[i] - 
#                                    list(Telengana['Recovered'])[i])


##########################################  West Bengal #####################33
West_Bengal = countries_day.loc[countries_day['State']=='West Bengal']
West_Bengal=West_Bengal.sort_values('Confirmed',ascending=True)

new_confirmed_cases_West_Bengal = []
new_confirmed_cases_West_Bengal.append( list(West_Bengal['Confirmed'])[0] - list(West_Bengal['Deaths'])[0] 
                           - list(West_Bengal['Recovered'])[0] )

for i in range(1,len(West_Bengal)):
    
    new_confirmed_cases_West_Bengal.append( list(West_Bengal['Confirmed'])[i] - 
                                     list(West_Bengal['Deaths'])[i] - 
                                    list(West_Bengal['Recovered'])[i])
############################################ 
p1 = figure(plot_width=800, plot_height=550, title="Trajectory of Covid-19")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Total number of detected cases (Log scale)'
p1.yaxis.axis_label = 'New confirmed cases (Log scale)'

p1.line(exponential_line_x, exponential_line_y, line_dash="4 4", line_width=0.5)

p1.line(np.log(list(Maharashtra['Confirmed'])), np.log(new_confirmed_cases_Maharashtra), color='#DBAE23', 
        legend_label='Maharashtra', line_width=1)
p1.circle(np.log(list(Maharashtra['Confirmed'])[-1]), np.log(new_confirmed_cases_Maharashtra[-1]), fill_color="white", size=5)

p1.line(np.log(list(Gujarat['Confirmed'])), np.log(new_confirmed_cases_Gujarat), color='#3EC358', 
        legend_label='Gujarat', line_width=1)
p1.circle(np.log(list(Gujarat['Confirmed'])[-1]), np.log(new_confirmed_cases_Gujarat[-1]), fill_color="white", size=5)

p1.line(np.log(list(Delhi['Confirmed'])), np.log(new_confirmed_cases_Delhi), color='#C3893E', 
       legend_label='Delhi', line_width=1)
p1.circle(np.log(list(Delhi['Confirmed'])[-1]), np.log(new_confirmed_cases_Delhi[-1]), fill_color="white", size=5)


p1.line(np.log(list(Madhya_Pradesh['Confirmed'])), np.log(new_confirmed_cases_Madhya_Pradesh), color='#3E4CC3', 
        legend_label='Madhya Pradesh', line_width=1)
p1.circle(np.log(list(Madhya_Pradesh['Confirmed'])[-1]), np.log(new_confirmed_cases_Madhya_Pradesh[-1]), fill_color="white", size=5)

p1.line(np.log(list(Rajasthan['Confirmed'])), np.log(new_confirmed_cases_Rajasthan), color='#F54138', 
        legend_label='Rajasthan', line_width=1)
p1.circle(np.log(list(Rajasthan['Confirmed'])[-1]), np.log(new_confirmed_cases_Rajasthan[-1]), fill_color="white", size=5)

p1.line(np.log(list(Uttar_Pradesh['Confirmed'])), np.log(new_confirmed_cases_Uttar_Pradesh), color='#23BCDB', 
        legend_label='Uttar Pradesh', line_width=1)
p1.circle(np.log(list(Uttar_Pradesh['Confirmed'])[-1]), np.log(new_confirmed_cases_Uttar_Pradesh[-1]), fill_color="white", size=5)

p1.line(np.log(list(Tamil_Nadu['Confirmed'])), np.log(new_confirmed_cases_Tamil_Nadu), color='#010A0C', 
        legend_label='Tamil Nadu', line_width=1)
p1.circle(np.log(list(Tamil_Nadu['Confirmed'])[-1]), np.log(new_confirmed_cases_Tamil_Nadu[-1]), fill_color="white", size=5)

p1.line(np.log(list(Andhra_Pradesh['Confirmed'])), np.log(new_confirmed_cases_Andhra_Pradesh), color='#bf40bf', 
        legend_label='Andhra Pradesh', line_width=1)
p1.circle(np.log(list(Andhra_Pradesh['Confirmed'])[-1]), np.log(new_confirmed_cases_Andhra_Pradesh[-1]), fill_color="white", size=5)

#p1.line(np.log(list(Telengana['Confirmed'])), np.log(new_confirmed_cases_Telengana), color='lime', 
#        legend_label='Telengana', line_width=1)
#p1.circle(np.log(list(Telengana['Confirmed'])[-1]), np.log(new_confirmed_cases_Telengana[-1]), fill_color="white", size=5)


p1.line(np.log(list(West_Bengal['Confirmed'])), np.log(new_confirmed_cases_West_Bengal), color='#0000ff', 
        legend_label='West Bengal', line_width=1)
p1.circle(np.log(list(West_Bengal['Confirmed'])[-1]), np.log(new_confirmed_cases_West_Bengal[-1]), fill_color="white", size=5)



p1.legend.location = "bottom_right"

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
p1 = figure(plot_width=800, plot_height=550, title="Trajectory of Covid-19")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Total number of detected cases (Log scale)'
p1.yaxis.axis_label = 'New confirmed cases (Log scale)'

p1.line(exponential_line_x, exponential_line_y, line_dash="4 4", line_width=0.5)

p1.line(np.log(list(Maharashtra['Confirmed'])), np.log(new_confirmed_cases_Maharashtra), color='#DBAE23', 
        legend_label='Maharashtra', line_width=1)
p1.circle(np.log(list(Maharashtra['Confirmed'])[-1]), np.log(new_confirmed_cases_Maharashtra[-1]), fill_color="white", size=5)

p1.line(np.log(list(West_Bengal['Confirmed'])), np.log(new_confirmed_cases_West_Bengal), color='Blue', 
        legend_label='West_Bengal', line_width=1)
p1.circle(np.log(list(West_Bengal['Confirmed'])[-1]), np.log(new_confirmed_cases_West_Bengal[-1]), fill_color="white", size=5)



p1.legend.location = "bottom_right"

output_file("coronavirus.html", title="coronavirus.py")

show(p1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from sklearn.metrics import mean_squared_error,r2_score
import statsmodels.api as sm
from datetime import timedelta
train_ml=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
valid_ml=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]
model_scores=[]

poly = PolynomialFeatures(degree = 8) 
train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
y=train_ml["Confirmed"]
linreg=LinearRegression(normalize=True)
linreg.fit(train_poly,y)
prediction_poly=linreg.predict(valid_poly)
rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))
model_scores.append(rmse_poly)
print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)          
comp_data=poly.fit_transform(np.array(india_datewise["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
predictions_poly=linreg.predict(comp_data)
plt.plot(india_datewise["Confirmed"],label="Train Confirmed Cases",linewidth=3)
plt.plot(india_datewise.index,predictions_poly, linestyle='--',label="Best Fit for Polynomial Regression",color='black')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases Polynomial Regression Prediction")
plt.xticks(rotation=90)
plt.legend()
new_date=[]
new_prediction_poly=[]
for i in range(1,18):
    new_date.append(india_datewise.index[-1]+timedelta(days=i))
    new_date_poly=poly.fit_transform(np.array(india_datewise["Days Since"].max()+i).reshape(-1,1))
    new_prediction_poly.append(linreg.predict(new_date_poly)[0])
model_predictions=pd.DataFrame(zip(new_date,new_prediction_poly),columns=["Date","Polynomial Regression Prediction"])
model_predictions
train_ml=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
valid_ml=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]
