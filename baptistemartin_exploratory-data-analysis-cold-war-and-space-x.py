import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd 

import numpy as np

import plotly.express as px

from iso3166 import countries

import plotly.graph_objects as go

import seaborn as sns

import matplotlib.pyplot as plt
space = pd.read_csv("/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv")

space.head(5)
#Modification Location

space["Country"] = space["Location"].map(lambda loc : loc.split(",")[-1].strip())

space["Launcher"] = space["Location"].map(lambda loc : loc.split(",")[0].strip())

space["Site name"] = space["Location"].map(lambda loc : loc.split(",")[1].strip())
# Wrong values 

sites = ['Xichang Satellite Launch Center','Imam Khomeini Spaceport','Blue Origin Launch Site',

         'Taiyuan Satellite Launch Center','Tai Rui Barge','Uchinoura Space Center',

         'Jiuquan Satellite Launch Center','Svobodny Cosmodrome']



#space["Launcher"] in sites

ind = []

for site in sites: 

    space["Site name"].iloc[space[space["Launcher"] == site].index] = site

    

space.replace('Alc?›ntara Launch Center','Alcantara Launch Center',inplace = True)

space.replace('M?\x81hia Peninsula','Mahia Peninsula',inplace = True)

#space['Site name'].unique() 
space["Country"].unique()
space["Site name"].replace('Kauai', 'Pacific Missile Range Facility',inplace = True)

space["Country"].replace('Yellow Sea', 'China',inplace = True)

space["Country"].replace('Shahrud Missile Test Site', 'Iran',inplace = True)

space["Country"].replace('Pacific Missile Range Facility', 'USA',inplace = True)

space["Country"].replace('Barents Sea', 'Russia',inplace = True)

space["Country"].replace('Pacific Ocean', 'USA',inplace = True)

space["Country"].replace('New Mexico', 'USA',inplace = True)

space["Country"].replace('Gran Canaria', 'Spain',inplace = True)



x = ["VKS RF","RVSN USSR","ROSCOSMOS"]

for c in x :

    space.loc[space['Company Name'] == c, "Country"] = "Russia"

del space["Unnamed: 0.1"]

del space["Location"]

del space["Launcher"]
#Year

space["Year"] = space["Datum"].map(lambda date : date.split(",")[-1]).map(lambda date : date.split(" ")[1])



# Day, month and Day number

space["Day"] = space["Datum"].map(lambda date : date.split(",")[0]).map(lambda day : day.split(" ")[0])

space["Month"] = space["Datum"].map(lambda date : date.split(",")[0]).map(lambda month : month.split(" ")[1])

space["Day Number"] = space["Datum"].map(lambda date : date.split(",")[0]).map(lambda day : day.split(" ")[2])



#Hour 

space["Hour"] = space["Datum"].map(lambda date : date.split(",")[-1])

Hours= []

for hour in space["Hour"]:

    if len(hour.split(" ")) > 2 :

        Hours.append(hour.split(" ")[2])

    else:

        Hours.append("")

space["Hour"] = Hours

space["Datum"] = pd.to_datetime(space["Datum"], utc = True)
space["Rocket Name"] = space["Detail"].map(lambda name : name.split("|")[1].strip())

space["Rocket Type"] = space["Detail"].map(lambda name : name.split("|")[0].strip())

del space["Detail"]
#space.reset_index(inplace = True)

space["Unnamed: 0"] = (space["Unnamed: 0"] + 1 - max(space["Unnamed: 0"]))*(-1)+2
#Geo analysist with scatter go 



country_alpha3 = {}

for c in countries:

    country_alpha3[c.name] = c.alpha3

    

space['alpha3'] = space['Country']

space = space.replace({"alpha3": country_alpha3})



space["alpha3"].unique()



#These countries have problem with the iso3166 norme, so I do the modification manually :

space.loc[space['Country'] == "North Korea", 'alpha3'] = "PRK"

space.loc[space['Country'] == "South Korea", 'alpha3'] = "KOR"

space.loc[space["Country"] == "Russia", 'alpha3'] = 'RUS'

space.loc[space["Country"] == "Iran", 'alpha3'] = "IRN"



space["alpha3"].unique()
space.rename(columns={" Rocket" : "Rocket price","Unnamed: 0" : "Launch Number"}, inplace = True)
# Price

space["Rocket price"] = space["Rocket price"].map(lambda x : str(x).strip()).map(lambda x : x.replace(",",""))

space["Rocket price"][1]

space["Rocket price"] = space["Rocket price"].map(lambda x : float(x))



# Hour / Day 



space["Hour"] = space["Hour"].map(lambda x : str(x).split(":")[0])

#I use -1 for the NaN values because np.Nan is only available for float type 

space["Hour"].replace("",-1,inplace = True)

space.Hour = space.Hour.astype('int32')

space["Day Number"] = space["Day Number"].astype('int64')

space.Year = space.Year.astype('int64')
space.head(5)
space.shape
space.tail(1)
space.describe(include="all")
fig = px.line(space, x='Datum', y="Launch Number",color_discrete_sequence = px.colors.sequential.RdBu, template = "plotly_dark",height = 600)

fig.show()
# Count_Year contain the sum of launches per year

Count_Year =space["Year"].value_counts().reset_index().rename(columns={"index" : "Year","Year" : "Count"}).sort_values("Year")

fig = px.bar(Count_Year, y='Count', x='Year', text='Count', color = 'Count',color_continuous_scale= px.colors.sequential.Reds, 

             template = "plotly_dark",height = 600)

fig.show()
#Count_Year_country

Count_Year_country = space.groupby(["Year","Country","alpha3"])["Launch Number"].count().reset_index()



fig = px.bar(Count_Year_country, y='Launch Number', x='Year',color="Country",

             color_discrete_sequence = px.colors.qualitative.Dark24,template = "plotly_dark"

             ,height = 600)

fig.show()
fig = px.scatter_geo(Count_Year_country,locations = "alpha3",

                     hover_name="Country", 

                     size="Launch Number",

                     color = "Launch Number",

                     animation_frame="Year",

                     projection="natural earth",

                     color_continuous_scale= px.colors.sequential.Reds,

                     template = "plotly_dark",

                     height = 600)

fig.show()
status = space["Status Mission"].value_counts().reset_index()

fig = px.pie(status, values='Status Mission', names='index',

             title='Success distribution', color_discrete_sequence = px.colors.sequential.RdBu, 

             template = "plotly_dark",height = 400)

             

fig.update_traces(textposition='outside', textinfo='percent+label')

fig.show()

Tab = space.groupby(["Year","Status Mission"])["Launch Number"].count().reset_index()

fig = px.bar(Tab, y='Launch Number', x='Year',color="Status Mission",title='Success distribution by Year',

             color_discrete_sequence = px.colors.sequential.Blackbody,template = "plotly_dark"

             ,height = 400)

fig.show()
x = space.groupby(["Country", "Status Mission"])["Launch Number"].count().reset_index()

x = x[~x.Country.isin(['USA','Russia'])]

fig = px.bar(x, x="Country", y="Launch Number", color="Status Mission",

             color_discrete_sequence = px.colors.sequential.Blackbody,

             template = "plotly_dark",

                  height=400)

fig.show()
x = space.groupby(["Country","Status Rocket"])["Launch Number"].count().reset_index().rename(columns = {"Launch Number" : "Count"})

fig = px.sunburst(x, 

                  path=['Status Rocket','Country'], 

                  values='Count',

                  color_discrete_sequence = px.colors.qualitative.Set1, 

                  template = "plotly_dark",

                  height=400)

fig.show()

fig = px.bar(x, x="Country", y="Count", color="Status Rocket", 

             color_discrete_sequence = px.colors.qualitative.Set1,

             template = "plotly_dark",

                  height=400)

fig.show()
x = space.groupby(["Country","Company Name"])["Launch Number"].count().reset_index().groupby(["Country"])["Company Name"].count().reset_index().sort_values("Company Name", ascending = True)

fig = px.bar(x, x="Company Name", 

             y="Country", 

             orientation='h',

             color_discrete_sequence = px.colors.sequential.RdBu,  

             template = "plotly_dark",

             height = 400)

fig.show()
company_grp = space.groupby(["Company Name", "Country"]).agg({"Rocket price" :"sum", "Launch Number" : "count"}).reset_index()

company_grp = company_grp[company_grp["Launch Number"] > 10].sort_values('Launch Number',ascending = True)

fig = px.bar(company_grp, 

             y='Company Name',

             x='Launch Number', 

             text='Launch Number',color = "Country",

             title = "Number of company by country", 

             template = "plotly_dark", 

             height = 600)

fig.show()
x = space.groupby(["Country","Site name"])["Launch Number"].count().reset_index()

x.head()

fig = px.sunburst(x, path=['Country', 'Site name'], values='Launch Number',

                  color_continuous_scale='RdBu',template = "plotly_dark")

fig.show()
x = space.groupby(["Month","Day"])["Launch Number"].count().reset_index()

x = x.pivot("Month","Day","Launch Number").fillna(0)

f, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(x, annot=True, fmt="d" ,linewidths=.5, ax=ax,cmap = "RdBu_r")
x = space.groupby(["Hour"])["Launch Number"].count().reset_index().sort_values("Hour")

fig = px.bar(x, x='Hour', y='Launch Number',orientation = 'v', color = 'Launch Number',color_continuous_scale='RdBu_r',template = "plotly_dark")

fig.show()
fig = px.histogram(space, x="Rocket price", marginal="box",template = "plotly_dark")

fig.show()
space[space["Rocket price"] == 5000.0]
#Création of the dataset

Cold_war = space[space.Country.isin(["USA","Russia"])]

Cold_war = Cold_war[Cold_war["Year"]<1992]
Cold_war.Datum.count()
#Count_Year_country

Count_Year_country = Cold_war.groupby(["Year","Country","alpha3"])["Launch Number"].count().reset_index()



fig = px.pie(Count_Year_country, values='Launch Number', names='Country',

             title='Launch distribution', color_discrete_sequence = px.colors.qualitative.Set1, 

             template = "plotly_dark",height = 600)

fig.show()



fig = px.bar(Count_Year_country, y='Launch Number', x='Year',color="Country", title = "Launches distribution per year",

             color_discrete_sequence = px.colors.qualitative.Set1, template = "plotly_dark"

             ,height = 600)

fig.show()
x = Cold_war.groupby(["Country","Company Name"])["Launch Number"].count().reset_index()

fig = px.sunburst(x, 

                  path=['Country','Company Name'], 

                  values='Launch Number', 

                  color_discrete_sequence = px.colors.qualitative.Set1,

                  title = "Repartion of companies per countries",

                  template = "plotly_dark",

                  height=600)

fig.show()
x = Cold_war.groupby(["Country","Rocket Type"])["Launch Number"].count().reset_index().sort_values("Launch Number",ascending=False)

print(x[x["Country"] == "Russia"].head(1))

print("With 388 launches, the Cosmos-3M were the rocket the most used by URSS")
print(x[x["Country"] == "USA"].head(1))

print("With 47 launches, the Atlas-SLV3 were the rocket the most used by USA")
usa = Cold_war[Cold_war["Country"] == "USA"].groupby(["Year","Status Mission"])["Launch Number"].count().reset_index()

urss = Cold_war[Cold_war["Country"] == "Russia"].groupby(["Year","Status Mission"])["Launch Number"].count().reset_index()

fig = px.bar(usa, y='Launch Number',title = "The USA",x='Year',color="Status Mission",

             color_discrete_sequence = px.colors.qualitative.Light24,template = "plotly_dark"

             ,height = 600)

fig.show()

fig = px.bar(urss, y='Launch Number', x='Year',color="Status Mission", title = "The URSS",

             color_discrete_sequence = px.colors.qualitative.Plotly,template = "plotly_dark"

             ,height = 600)

fig.show()
x = Cold_war[Cold_war["Status Mission"].isin(["Failure", "Partial Failure", "Prelaunch Failure"])].groupby("Country")["Launch Number"].count().reset_index()

fig = px.bar(x, y='Launch Number', x='Country',color = "Country",color_discrete_sequence = px.colors.qualitative.Set1,template = "plotly_dark"

             ,height = 400)

fig.show()
Years = usa.Year.unique()

percentage_usa = []

percentage_urss = []

launch = usa["Launch Number"]

for y in Years:

    ind = usa[usa["Year"] == int(y)].index

    

    if len(ind) == 2:       

        result = launch[ind[1]]/(launch[ind[0]] + launch[ind[1]])

    elif len(ind) == 3:

        result = launch[ind[2]]/(launch[ind[0]] + launch[ind[1]] + launch[ind[2]])

    elif len(ind) == 1:

        if usa["Status Mission"].iloc[ind[0]] == "Failure":

            result = 0

        else: 

            result = 1 

    percentage_usa.append(result*100)



launch = urss["Launch Number"]

for y in Years:

    ind = urss[urss["Year"] == int(y)].index

    

    if len(ind) == 2:       

        result = launch[ind[1]]/(launch[ind[0]] + launch[ind[1]])

    elif len(ind) == 3:

        result = launch[ind[2]]/(launch[ind[0]] + launch[ind[1]] + launch[ind[2]])

    elif len(ind) == 4:

        result = launch[ind[3]]/(launch[ind[0]] + launch[ind[1]] + launch[ind[2]] + launch[ind[3]])

    elif len(ind) == 1:

        result = 1 

    percentage_urss.append(result*100)



Evo = pd.DataFrame(Years, columns = ["Year"])

Evo["Percentage_usa"] = percentage_usa

Evo["Percentage_urss"] = percentage_urss



fig = go.Figure()

fig.add_trace(go.Scatter(x=Evo["Year"], y=Evo["Percentage_usa"], name="USA"))

fig.add_trace(go.Scatter(x=Evo["Year"], y=Evo["Percentage_urss"], name="URSS"))



fig.show()
Count_Year_country = Count_Year_country.sort_values(["Year","Launch Number"],ascending = False)

Count_Year_country2 = pd.concat([country[1].head(1) for country in Count_Year_country.groupby(["Year"])])



p = []

c = []

for y in Evo.Year: 

    p_usa = float(Evo[Evo["Year"]== y]["Percentage_usa"])

    p_urss = float(Evo[Evo["Year"]== y]["Percentage_urss"])

    if p_usa > p_urss :

        p.append(p_usa)

        c.append("USA")

    else:

        p.append(p_urss)

        c.append("URSS")



Evo["Max"] = p

Evo["Country"] = c



fig = px.bar(Count_Year_country2, y='Launch Number', x='Year',color="Country", title = "Launch Number",

             color_discrete_sequence = px.colors.qualitative.Set1,template = "plotly_dark"

             ,height = 600)

fig.show()

fig = px.bar(Evo, y='Max', x='Year',color="Country", title = "Percentage of successfull mission",

             color_discrete_sequence = px.colors.qualitative.Set1,template = "plotly_dark"

             ,height = 600)

fig.show()
print(365/(Cold_war[Cold_war["Country"] == "Russia"]["Launch Number"].count()/34))

print(365/(Cold_war[Cold_war["Country"] == "USA"]["Launch Number"].count()/34))

spacex = space[space["Company Name"] == "SpaceX"]

spacex.shape
x = spacex.groupby(["Year"])["Launch Number"].count().reset_index()

fig = px.line(x, x='Year', y="Launch Number",color_discrete_sequence = px.colors.qualitative.D3, template = "plotly_dark",height = 600)

fig.show()
x = spacex.groupby(["Status Mission"])["Launch Number"].sum().reset_index()

fig = px.pie(x, values="Launch Number", names ='Status Mission', color_discrete_sequence = px.colors.qualitative.G10, 

             template = "plotly_dark",height = 600)

fig.show()
x = spacex.groupby(["Year","Status Mission"])["Launch Number"].count().reset_index().sort_values(["Launch Number"],ascending = False)

fig = px.bar(x, y='Launch Number', x='Year',color = "Status Mission", title = "Launches repartion per years",

              color_discrete_sequence = px.colors.qualitative.G10, template = "plotly_dark"

             ,height = 600)

fig.show()
sumprice = spacex["Rocket price"].sum()

print("To start easily, we can see than Space X has invested approximately a total of {} millions dollars in its rocket".format(sumprice))
x = spacex.groupby(["Year"])["Rocket price"].sum().reset_index()

fig = px.bar(x, y='Rocket price', x='Year',color = "Rocket price",

              color_continuous_scale='Blues', template = "plotly_dark"

             ,height = 600)

fig.show()
x = spacex[spacex["Status Mission"] == "Success"].groupby(["Status Mission","Status Rocket"]).agg({"Launch Number": "count", "Rocket price" : "sum"}).reset_index()

fig = px.bar(x, y='Status Rocket', x='Launch Number',color = "Rocket price",

              color_continuous_scale='Blues', template = "plotly_dark"

             ,height = 600, orientation='h')

fig.show()
fig = px.histogram(spacex, x="Rocket price", marginal="box", template = "plotly_dark"

             ,height = 600,hover_data=spacex.columns)

fig.show()
x = spacex.groupby(["Rocket Type"]).agg({"Launch Number": "count", "Rocket price" : "mean"}).reset_index().sort_values(["Launch Number"], ascending = False)

fig = px.bar(x, y='Launch Number', x='Rocket Type',color = "Rocket price",

              color_continuous_scale='Blues', template = "plotly_dark"

             ,height = 600)

fig.show()