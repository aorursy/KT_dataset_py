import pandas as pd 
df=pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv',parse_dates=['Date'])

df.head()
df=df.rename(columns={"Country/Region": "Country"})

df.Country=df["Country"].astype(str)

df_country_all=df.groupby([ "Date","Country"])[["Date", "Country", "Confirmed", "Deaths", "Recovered","Lat","Long"]].sum().sort_values(["Date","Country"]).reset_index()

df_country_all.Date=df_country_all["Date"].astype(str)

#df_country=df_country.query("Country==['Italy','US','China','Iran','Spain','France']")

#df_country=df_country[df_country['Date']==df_country.Date.max() & df_country['Confirmed'] > 15000]

df_country_all=df_country_all[df_country_all.Country.isin(df_country_all[(df_country_all['Date']==df_country_all.Date.max())].Country.unique())]

df_country_all["Combined"]=df_country_all["Country"]+ ":" + df_country_all["Confirmed"].astype(str)

df_country_latest=df_country_all[df_country_all.Date==df_country_all.Date.max()]
df_country=df.groupby([ "Date","Country"])[["Date", "Country", "Confirmed", "Deaths", "Recovered"]].sum().sort_values(["Date","Country"]).reset_index()

df_country.Date=df_country["Date"].astype(str)

#df_country=df_country.query("Country==['Italy','US','China','Iran','Spain','France']")

#df_country=df_country[df_country['Date']==df_country.Date.max() & df_country['Confirmed'] > 15000]

df_country=df_country[df_country.Country.isin(df_country[(df_country['Date']==df_country.Date.max()) & (df_country['Confirmed'] > 10000)].Country.unique())]

df_country["Combined"]=df_country["Country"]+ ":" + df_country["Confirmed"].astype(str)
import plotly.express as px

fig = px.pie(df_country_latest, values='Confirmed', names='Country',title="% Of Confirmed Cases by Country")

fig.update_traces(textposition='inside', textfont_size=14)

fig.show()
fig = px.line(df_country, x="Date", y="Confirmed", color="Country",

              line_group="Country", hover_name="Country",title="Trend Of Confirmed Cases For Countries with > 10K Cases")

fig.show()




fig = px.scatter(df_country, x="Confirmed", y="Deaths", animation_frame="Date", animation_group="Country",

           size="Confirmed", color="Country", hover_name="Country", text="Combined",

        size_max=60, range_x=[300,1000000], range_y=[-100,12000],title="Covid-19 Confirmed Vs Deaths - Countries with cases > 10K (Click on the Play button to see the animation of growing trend)",opacity=0.8,log_x=True,height=850)

fig.layout.update(showlegend=False,xaxis_showgrid=False, yaxis_showgrid=False) 

fig.layout.plot_bgcolor = 'White'

fig.update_traces(textposition='top center')

fig.show()


fig = px.scatter_geo(df_country_all, locations="Country", color="Country", hover_name="Country", size="Confirmed",text="Combined",

               animation_frame="Date", projection="kavrayskiy7",locationmode="country names",title="Geographical View of Confirmed Cases - All Countries")

fig.layout.update(showlegend=False) 

#fig.update_traces(textposition='top center')

fig.show()
fig = px.scatter(df_country_latest, x="Confirmed", y="Deaths",  

           size="Confirmed", color="Country", hover_name="Country",

           log_x=True, size_max=45,title="Covid-19 Confirmed Vs Deaths - All Countries")

fig.show()
fig = px.histogram(df_country_latest[df_country_latest.Confirmed < 20000], x="Confirmed",title="Distribution Of Confirmed Cases for Countries with < 20K cases")

fig.show()