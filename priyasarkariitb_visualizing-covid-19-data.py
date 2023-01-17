import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import ticker 



import folium

from datetime import datetime,date

from scipy.interpolate import make_interp_spline, BSpline



%matplotlib inline
!pip install folium

!pip install calmap
#### Reading the Timeseries data 



df_confirmed=pd.read_csv('../input/covid19-timeseries-data/confirmed_global.csv')

df_deaths = pd.read_csv('../input/covid19-timeseries-data/deaths_global.csv')

df_recovered=pd.read_csv('../input/covid19-timeseries-data/recovered_global.csv')
print(df_confirmed.head(5))

print(df_deaths.head(5))

print(df_recovered.head(5))
######## Rename the Columns ###############

df_confirmed = df_confirmed.rename(columns={"Province/State":"state","Country/Region": "country"})

df_deaths = df_deaths.rename(columns={"Province/State":"state","Country/Region": "country"})

df_recovered = df_recovered.rename(columns={"Province/State":"state","Country/Region": "country"})



######## Pre Process the Data ##############

# Changing the conuntry names as required by pycountry_convert Lib

df_confirmed.loc[df_confirmed['country'] == "US", "country"] = "USA"

df_deaths.loc[df_deaths['country'] == "US", "country"] = "USA"

df_recovered.loc[df_recovered['country'] == "US", "country"] = "USA"



df_confirmed.loc[df_confirmed['country'] == 'Korea, South', "country"] = 'South Korea'

df_deaths.loc[df_deaths['country'] == 'Korea, South', "country"] = 'South Korea'

df_recovered.loc[df_recovered['country'] == 'Korea, South', "country"] = 'South Korea'



df_confirmed.loc[df_confirmed['country'] == 'Taiwan*', "country"] = 'Taiwan'

df_deaths.loc[df_deaths['country'] == 'Taiwan*', "country"] = 'Taiwan'

df_recovered.loc[df_recovered['country'] == 'Taiwan*', "country"] = 'Taiwan'



df_confirmed.loc[df_confirmed['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'

df_deaths.loc[df_deaths['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'

df_recovered.loc[df_recovered['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'



df_confirmed.loc[df_confirmed['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"

df_deaths.loc[df_deaths['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"

df_recovered.loc[df_recovered['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"



df_confirmed.loc[df_confirmed['country'] == "Reunion", "country"] = "Réunion"

df_deaths.loc[df_deaths['country'] == "Reunion", "country"] = "Réunion"

df_recovered.loc[df_recovered['country'] == "Reunion", "country"] = "Réunion"



df_confirmed.loc[df_confirmed['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'

df_deaths.loc[df_deaths['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'

df_recovered.loc[df_recovered['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'



df_confirmed.loc[df_confirmed['country'] == 'Bahamas, The', "country"] = 'Bahamas'

df_deaths.loc[df_deaths['country'] == 'Bahamas, The', "country"] = 'Bahamas'

df_recovered.loc[df_recovered['country'] == 'Bahamas, The', "country"] = 'Bahamas'



df_confirmed.loc[df_confirmed['country'] == 'Gambia, The', "country"] = 'Gambia'

df_deaths.loc[df_deaths['country'] == 'Gambia, The', "country"] = 'Gambia'

df_recovered.loc[df_recovered['country'] == 'Gambia, The', "country"] = 'Gambia'



####### Handling Missing Values #######

df_confirmed = df_confirmed.replace(np.nan, '', regex=True)

df_deaths = df_deaths.replace(np.nan, '', regex=True)

df_recovered = df_recovered.replace(np.nan, '', regex=True)
####### Converting TimeSeries Data into Totals ##########

####### Adding a Column to all the 3 dataset which is basically total number of cases till date #####

####### Get sum of a particular row : https://www.geeksforgeeks.org/python-pandas-dataframe-sum/

####### Access particular columns in Dataframe use df.iloc[row_n:row_l,col_n:col_l] https://stackoverflow.com/questions/17193850/how-to-get-column-by-number-in-pandas

df_confirmed["total_confirmed"]=df_confirmed.iloc[:,4:].sum(axis = 1, skipna = True) 

df_deaths["total_deaths"]=df_deaths.iloc[:,4:].sum(axis = 1, skipna = True) 

df_recovered["total_recovered"]=df_recovered.iloc[:,4:].sum(axis = 1, skipna = True) 
###### Now Let's create a New DataFrames which consist of country name, total confirmed cases, total deaths and total recovered

###### Created by merging the 3 datasets with reference to their country names

df_net=pd.DataFrame({"state":df_confirmed["state"],"country":df_confirmed["country"],"confirmed":df_confirmed["total_confirmed"],"deaths":df_deaths["total_deaths"],"recovered":df_recovered["total_recovered"]})





###### GroupBy for Country wise data

df_country=df_net.groupby("country").agg(total_confirmed=pd.NamedAgg(column='confirmed', aggfunc='sum'),

                              total_death=pd.NamedAgg(column='deaths', aggfunc='sum'),

                              total_recovered=pd.NamedAgg(column='recovered', aggfunc='sum'))



####### Lets create a new column to our new dataframes total_active_cases cases which gives us an idea about number of active cases 

df_country["active"]=df_country["total_confirmed"]-df_country["total_recovered"]-df_country["total_death"]

df_country.head()
#Top 10 countries (Confirmed Cases and Deaths)

f = plt.figure(figsize=(10,5))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.barh(df_country.sort_values('total_confirmed')["total_confirmed"].index[-10:],df_country.sort_values('total_confirmed')["total_confirmed"].values[-10:],color="darkcyan")

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Confirmed Cases",fontsize=18)

plt.title("Top 10 Countries (Confirmed Cases)",fontsize=20)

plt.grid(alpha=0.3)



f = plt.figure(figsize=(10,5))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.barh(df_country.sort_values('total_death')["total_death"].index[-10:],df_country.sort_values('total_death')["total_death"].values[-10:],color="red")

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Deaths",fontsize=18)

plt.title("Top 10 Countries (Deaths)",fontsize=20)

plt.grid(alpha=0.3)



f = plt.figure(figsize=(10,5))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.barh(df_country.sort_values('total_recovered')["total_recovered"].index[-10:],df_country.sort_values('total_recovered')["total_recovered"].values[-10:],color="green")

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Recovered Cases",fontsize=18)

plt.title("Top 10 Countries (Recovered Cases)",fontsize=20)

plt.grid(alpha=0.3)



# Visualization on World Map

world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2,max_zoom=6)

for i in range(0,len(df_confirmed)):

    folium.Circle(

        location=[df_confirmed.iloc[i]['Lat'], df_confirmed.iloc[i]['Long']],

        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df_confirmed.iloc[i]['country']+"</h5>"+

                    "<div style='text-align:center;'>"+str(np.nan_to_num(df_confirmed.iloc[i]['state']))+"</div>"+

                    "<hr style='margin:10px;'>"+

                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

        "<li>Confirmed: "+str(df_confirmed.iloc[i,-1])+"</li>"+

        "<li>Deaths:   "+str(df_deaths.iloc[i,-1])+"</li>"+

        "<li>Mortality Rate:   "+str(np.round(df_deaths.iloc[i,-1]/(df_confirmed.iloc[i,-1]+1.00001)*100,2))+"</li>"+

        "</ul>"

        ,

        radius=(int((np.log(df_confirmed.iloc[i,-1]+1.00001)))+0.2)*50000,

        color='#ff6600',

        fill_color='#ff8533',

        fill=True).add_to(world_map)



world_map

import plotly.express as px

# Heat Maps of Death & Recovery

temp_df = pd.DataFrame(df_country['total_death'])

temp_df = temp_df.reset_index()

fig = px.choropleth(temp_df, locations="country",

                    color=np.log10(temp_df["total_death"]), # lifeExp is a column of gapminder

                    hover_name="country", # column to add to hover information

                    hover_data=["total_death"],

                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(title_text="Total Deaths Heat Map (Log Scale)")

fig.update_coloraxes(colorbar_title="Total Death(Log Scale)",colorscale="Reds")



fig.show()



temp_df = pd.DataFrame(df_country['total_recovered'])

temp_df = temp_df.reset_index()

fig = px.choropleth(temp_df, locations="country",

                    color=np.log10(temp_df["total_recovered"]), # lifeExp is a column of gapminder

                    hover_name="country", # column to add to hover information

                    hover_data=["total_recovered"],

                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(title_text="Recovered Cases Heat Map (Log Scale)")

fig.update_coloraxes(colorbar_title="Recovered Cases(Log Scale)",colorscale="Greens")



fig.show()
###### Transforming Data For Visualizing Spread ######

df_confirmed=df_confirmed.drop("total_confirmed",1)

df_confirmed_T=df_confirmed.melt(id_vars=["state","country","Lat","Long"], 

                var_name="Date", 

                value_name="Confirmed")



df_deaths=df_deaths.drop("total_deaths",1)

df_deaths_T=df_deaths.melt(id_vars=["state","country","Lat","Long"], 

                var_name="Date", 

                value_name="Deaths")



df_recovered=df_recovered.drop("total_recovered",1)

df_recovered_T=df_recovered.melt(id_vars=["state","country","Lat","Long"], 

                var_name="Date", 

                value_name="Recovered")



df_net_T=pd.DataFrame({"country":df_confirmed_T["country"],"date":df_confirmed_T["Date"],"Confirmed":df_confirmed_T["Confirmed"],"Deaths":df_deaths_T["Deaths"],"Recovered":df_recovered_T["Recovered"]})



df_net_T["date"]=pd.to_datetime(df_net_T["date"])
df_net_global=df_net_T.groupby("date").agg({"Confirmed":sum,

                                           "Deaths":sum,

                                           "Recovered":sum})
#### Total Spread all over the Globe

df_net_global["Confirmed"] = pd.to_numeric(df_net_global["Confirmed"]).fillna(0)

df_net_global["Deaths"] = pd.to_numeric(df_net_global["Deaths"]).fillna(0)

df_net_global["Recovered"] = pd.to_numeric(df_net_global["Recovered"]).fillna(0)



f,ax = plt.subplots(figsize=(20,5))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.scatter(df_net_global.index,np.cumsum(df_net_global["Confirmed"]),color="blue")

plt.scatter(df_net_global.index,np.cumsum(df_net_global["Deaths"]),color="red")

plt.scatter(df_net_global.index,np.cumsum(df_net_global["Recovered"]),color="green")



plt.xticks(rotation=90,fontsize=10)

plt.xlabel("Confirmed Cases",fontsize=18)

plt.title("Top 10 Countries (Confirmed Cases)",fontsize=20)

plt.grid(alpha=0.3)
# Spread Trend per country

df_net_T["Confirmed"] = pd.to_numeric(df_net_T["Confirmed"]).fillna(0)

df_net_T["Deaths"] = pd.to_numeric(df_net_T["Deaths"]).fillna(0)

df_net_T["Recovered"] = pd.to_numeric(df_net_T["Recovered"]).fillna(0)



f,ax = plt.subplots(figsize=(20,5))

f.add_subplot(111)

plt.axes(axisbelow=True)



countries=df_country.sort_values('total_confirmed')["total_confirmed"].index[-10:]



for country in countries:

    df=df_net_T[df_net_T["country"]==country]

    plt.scatter(df["date"],np.cumsum(df["Confirmed"]),c=np.random.rand(3,),label=country)



plt.legend(loc="upper right")

plt.xticks(rotation=90,fontsize=10)

plt.xlabel("Confirmed Cases",fontsize=18)

plt.title("Top 10 Countries (Confirmed Cases)",fontsize=20)

plt.grid(alpha=0.3)
# Calendar Spread Analysis

import calmap



f = plt.figure(figsize=(20,10))

f.add_subplot(2,1,1)

calmap.yearplot(df_net_global["Confirmed"], year=2020)
# Racer maps of Spread

df_net_T["date"]=df_net_T["date"].astype(str)

fig = px.scatter_geo(df_net_T,locations="country", locationmode='country names', 

                     color=np.power(df_data["Confirmed"],0.3)-2 , size= np.power(df_data["Confirmed"]+1,0.3)-1, hover_name="country",

                     hover_data=["Confirmed"],

                     range_color= [0, max(np.power(df_data["Confirmed"],0.3))], 

                     projection="natural earth", animation_frame="date", 

                     color_continuous_scale=px.colors.sequential.Plasma,

                     title='COVID-19: Progression of spread'

                    )

fig.update_coloraxes(colorscale="hot")

fig.update(layout_coloraxis_showscale=False)

fig.show()