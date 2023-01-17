import numpy as np 

import matplotlib.pyplot as plt 

import plotly.express as px

import plotly.graph_objs as go

import matplotlib.colors as mcolors

import pandas as pd 

import seaborn as sns

import datetime

import operator 

plt.style.use('fivethirtyeight')

%matplotlib inline 
mal_data = pd.read_csv("../input/coronavirus-malaysia/UKK_ CoViD-19 Current Situation in Malaysia__Main_Table.csv")

combo_chart = pd.read_csv("../input/coronavirus-malaysia/UKK_ CoViD-19 Current Situation in Malaysia__Main_Combo chart.csv")

cases_type = pd.read_csv("../input/coronavirus-malaysia/covid-19-my-cases-types.csv")

covid_mal = pd.read_csv("../input/coronavirus-malaysia/covid-19-malaysia.csv")

state_cases = pd.read_csv("../input/coronavirus-malaysia/covid-19-my-states-cases.csv")
print(mal_data.shape) #16 State	Confirmed	New Cases	Deaths	Population ('000

print(combo_chart.shape) #Date	States	No.of Cases

print(cases_type.shape)#date	pui	close-contact	tabligh	surveillance	hadr	import

print(covid_mal.shape)

print(state_cases.shape)
mal_data.head(10)
combo_chart.head(10)
cases_type.head(10)
state_cases.head(10)
mal_data.info()
combo_chart.info()
cases_type.info()
state_cases.info()
state_cases["total cases"] = state_cases.sum(axis=1)

state_cases
gradient_map = mal_data.groupby("State")["Confirmed"].sum().sort_values(ascending = False).to_frame()

gradient_map.style.background_gradient(cmap="Reds")
gradient_map = mal_data.groupby("State")["New Cases"].sum().sort_values(ascending = False).to_frame()

gradient_map.style.background_gradient(cmap="Reds")
# Data from UKK : Confirmed cases in Malaysia and new cases according to state
mal_data
fig = px.bar(mal_data.sort_values("Confirmed", ascending = False).sort_values("Confirmed", ascending = True),

            x = "Confirmed", y = "State",orientation = 'h', title = "Confirmed cases in Malaysia(UKK DOSM)", text = "Confirmed", width =1000,

            height=700, range_x = [0, max(mal_data["Confirmed"])])



fig.update_layout(plot_bgcolor = "rgb(250, 242, 242)")

fig.show()
fig = px.bar(mal_data.sort_values("New Cases", ascending = False).sort_values("New Cases", ascending = True),

            x = "New Cases", y = "State",orientation = 'h', title = "New Cases According to State(UKK DOSM)", text = "New Cases", width =800,

            height=1000, range_x = [0, max(mal_data["New Cases"])])



fig.update_layout(plot_bgcolor = "rgb(250, 242, 242)")

fig.show()
k =state_cases.drop(state_cases.index[2])
fig = go.Figure()

fig.add_trace(go.Scatter(x = k["date"], y = k["perlis"], mode = "lines", name ="Perlis"))

fig.add_trace(go.Scatter(x = k["date"], y = k["kedah"], mode = "lines", name ="Kedah"))

fig.add_trace(go.Scatter(x = k["date"], y = k["pulau-pinang"], mode = "lines", name ="Pulau Pinang"))

fig.add_trace(go.Scatter(x = k["date"], y = k["perak"], mode = "lines", name ="Perak"))

fig.add_trace(go.Scatter(x = k["date"], y = k["selangor"], mode = "lines", name ="Selangor"))

fig.add_trace(go.Scatter(x = k["date"], y = k["negeri-sembilan"], mode = "lines", name ="Negari Sembilan"))

fig.add_trace(go.Scatter(x = k["date"], y = k["melaka"], mode = "lines", name ="Melaka"))

fig.add_trace(go.Scatter(x = k["date"], y = k["johor"], mode = "lines", name ="Johor"))

fig.add_trace(go.Scatter(x = k["date"], y = k["pahang"], mode = "lines", name ="Pahang"))

fig.add_trace(go.Scatter(x = k["date"], y = k["terengganu"], mode = "lines", name ="Terengganu"))

fig.add_trace(go.Scatter(x = k["date"], y = k["kelantan"], mode = "lines", name ="Kelantan"))

fig.add_trace(go.Scatter(x = k["date"], y = k["sabah"], mode = "lines", name ="Sabah"))

fig.add_trace(go.Scatter(x = k["date"], y = k["sarawak"], mode = "lines", name ="Sarawak"))

fig.add_trace(go.Scatter(x = k["date"], y = k["wp-kuala-lumpur"], mode = "lines", name ="Wp Kuala Lumpur"))

fig.add_trace(go.Scatter(x = k["date"], y = k["wp-putrajaya"], mode = "lines", name ="Wp Putrajaya"))

fig.add_trace(go.Scatter(x = k["date"], y = k["wp-labuan"], mode = "lines", name ="Wp Putrajaya"))



fig.update_layout(title_text = "Trend of Corona Virus Cases in Malaysia(State)", plot_bgcolor = 'rgb(250,242,242)')



fig.show()
covid_mal["date"] = pd.to_datetime(covid_mal["date"], dayfirst = True)
covid_mal["end of month"] = covid_mal["date"].dt.is_month_end

covid_mal
q = covid_mal.index[covid_mal["end of month"]]

flt_covid=covid_mal.loc[q]

flt_covid
flt_covid['date']=flt_covid['date'].astype(str)
f, ax = plt.subplots(figsize = (12, 5))

data = flt_covid[["date", "cases", "discharged", "death"]]

# data.sort_values("cases", inplace = True)

sns.set_color_codes("pastel")

sns.barplot(x = "cases", y = "date", data=data, label="Cases", color = "b")

sns.barplot(x = "discharged", y = "date", data=data, label="Recover", color = "g")

sns.barplot(x = "death", y = "date", data=data, label="Death", color = "r")



# Add legend

ax.legend(ncol=2, loc = "lower right", frameon = True)

ax.set(xlim =(0, 10000), ylabel = "", xlabel = "Cases")

ax.title.set_text("Corona Virus Total, Recover and Death Cases in Malaysia")

sns.despine(left = True, bottom = True)
fig = go.Figure()

fig.add_trace(go.Scatter(x = covid_mal["date"], y = covid_mal["cases"], mode = "lines", name ="Total Cases"))





fig.update_layout(title_text = "Trend of Corona Virus Cases in Malaysia(Cumulative Cases)", plot_bgcolor = 'rgb(250,242,242)')



fig.show()
k["new cases"] = k["total cases"].diff()

k
covid_mal["New Cases"] = covid_mal["cases"].diff()

covid_mal = covid_mal.fillna(0)

covid_mal 
fig = go.Figure()

fig = px.bar(covid_mal, x="date", y="New Cases", barmode='group',

             height=400)

fig.update_layout(title_text='New Coronavirus Cases in Malaysia (per day)',plot_bgcolor='rgb(250, 242, 242)', xaxis_title = "")



fig.show()
covid_mal["mortality rate"] = ((covid_mal["death"]/covid_mal["cases"])*100).round(3)

covid_mal["recover rate"] = ((covid_mal["discharged"]/covid_mal["cases"])*100).round(3)

covid_mal.tail(10)
fig = go.Figure()

fig.add_trace(go.Scatter(x = covid_mal["date"], y = covid_mal["recover rate"], mode = "lines", name ="Recovery (Recover/Cases)"))

fig.add_trace(go.Scatter(x = covid_mal["date"], y = covid_mal["mortality rate"], mode = "lines", name ="Mortality (Death/Cases)"))





fig.update_layout(title_text = " Malaysia Corona Virus Recovery and Mortality Rate", plot_bgcolor = 'rgb(250,242,242)',yaxis_title= "% percentage")



fig.show()
k.columns.values
fig = go.Figure()

fig = px.bar(k, x="date", y=k["pulau-pinang"].diff(), barmode='group',

             height=400)

fig.update_layout(title_text='Pulau-Pinang',plot_bgcolor='rgb(250, 242, 242)', yaxis_title = "New Cases")



fig.show()
fig = go.Figure()

fig = px.bar(k, x="date", y=k["perak"].diff(), barmode='group',

             height=400)

fig.update_layout(title_text='Perak',plot_bgcolor='rgb(250, 242, 242)', yaxis_title = "New Cases")



fig.show()
fig = go.Figure()

fig = px.bar(k, x="date", y=k["selangor"].diff(), barmode='group',

             height=400)

fig.update_layout(title_text='Selangor',plot_bgcolor='rgb(250, 242, 242)', yaxis_title = "New Cases")



fig.show()
fig = go.Figure()

fig = px.bar(k, x="date", y=k["wp-kuala-lumpur"].diff(), barmode='group',

             height=400)

fig.update_layout(title_text='Wp Kuala Lumpur',plot_bgcolor='rgb(250, 242, 242)', yaxis_title = "New Cases")



fig.show()
fig = go.Figure()

fig = px.bar(k, x="date", y=k["johor"].diff(), barmode='group',

             height=400)

fig.update_layout(title_text='Johor',plot_bgcolor='rgb(250, 242, 242)', yaxis_title = "New Cases")



fig.show()
mal_data
state = mal_data["State"].values

state
import folium
data = pd.DataFrame({

   'lat':[3.166665872, 3.066695996, 2.710492166, 2.033737609, 3.16640749, 4.18400112, 5.046396097, 4.01185976, 2.206414407, 6.119973978, 5.417071146, 4.233241596, 2.914019795, 5.649718444, 6.433001991,5.2770],

   'lon':[101.6999833, 101.5499977, 101.9400203, 102.566597, 113.0359838, 102.0420006, 118.3359704, 101.0314453, 102.2464615, 102.2299768, 100.4000109, 103.4478869, 101.701947, 100.4793343,100.1899987,115.2436],

   'name':state,

   'value':[2028, 1909,  858,  675,  552,  361,  346,  256,  219,  156,  121,

        111,   97,   96,   18,   16]

})

data
malaysia_location = [4.2105, 101.9758]

m = folium.Map(location=malaysia_location, zoom_start=7, tiles = "OpenStreetMap")



for lat, lon, name, value in zip(data['lat'], data['lon'], data['name'], data['value']):

    folium.Circle([lat, lon],radius=(value*20),popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Active Cases</strong>: ' + str(value) + '<br>'),color='red',fill_color='red',fill_opacity=0.3 ).add_to(m)



m