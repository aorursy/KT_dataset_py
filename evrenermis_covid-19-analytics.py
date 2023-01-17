import numpy as np 

import pandas as pd



df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", parse_dates = ["Last Update"])

df_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_death = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

df_recovered= pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_coordinates = pd.read_csv("../input/world-coordinates/world_coordinates.csv")

df_plotly_country_codes = pd.read_csv("../input/plotlycountrycodes/plotly_countries_and_codes.csv")
def data_preprocessing(data):

    # Generic Data PreProcessing Steps

    data.columns = data.columns.str.replace(" ", "_")

    data.columns = map(str.lower, data.columns)

    return data



# df Pre-Processings

df = data_preprocessing(df)

df.rename(columns={"country/region":"country"},inplace=True)

df["country"].replace({"Mainland China":"China", "US":"United States", 

                       "UK":"United Kingdom"}, inplace=True)

df["country"].replace(to_replace = "US", 

                      value = "United States", inplace=True) 

df["country"].replace(to_replace = "UK", 

                      value = "United Kingdom", inplace=True) 



# df_coordinates Pre-Processings

df_coordinates = data_preprocessing(df_coordinates)

df_coordinates["country"].replace(to_replace = "US", 

                                  value = "United States", inplace=True)

df_coordinates["country"].replace(to_replace = "UK", 

                                  value = "United Kingdom", inplace=True)



# df_plotly_country_codes Pre-Processing

df_plotly_country_codes = data_preprocessing(df_plotly_country_codes)



# Other df Pre-Processing

df_confirmed = data_preprocessing(df_confirmed)

df_confirmed.rename(columns={"country/region":"country"},inplace=True)

df_confirmed.drop(columns={"province/state", "lat", "long"}, inplace=True)

df_death = data_preprocessing(df_death)

df_death.rename(columns={"country/region":"country"},inplace=True)

df_death.drop(columns={"province/state", "lat", "long"}, inplace=True)

df_recovered = data_preprocessing(df_recovered)

df_recovered.rename(columns={"country/region":"country"},inplace=True)

df_recovered.drop(columns={"province/state", "lat", "long"}, inplace=True)

# ----------- df_final set------------------------------------------------------------------------------------------------------------------------------------------------

df.sort_values("last_update", ascending=False, inplace=True)

df["last_update_str"] = df["last_update"].dt.date.astype(str)

df["active"] = df["confirmed"] - (df["deaths"] + df["recovered"])

df_final = df[df["last_update_str"] == "2020-03-19"]



# ----------- df_table for below graphs-----------------------------------------------------------------------------------------------------------------------------

df_table = df_final.groupby(["country", "last_update_str"]).sum().reset_index().sort_values("confirmed", ascending=False).drop(columns={"sno"})

df_table["confirmed_ratio"] = round(df_table['confirmed'] / df_table["confirmed"].sum() * 100,0)

df_table["deaths_ratio"] = round(df_table['deaths'] / df_table["deaths"].sum() * 100,0)

df_table["recovered_ratio"] = round(df_table['recovered'] / df_table["recovered"].sum() * 100,0)

df_table["active_ratio"] = round(df_table['active'] / df_table["active"].sum() * 100,0)

df_table = df_table.merge(df_coordinates, left_on="country", right_on="country")

df_table.drop(columns="code", inplace=True)

df_table["text"] = "Country=" + df_table["country"].map(str) + " " + "Confirmed=" + df_table["confirmed"].map(str) + " " + "Deaths=" + df_table["deaths"].map(str)

df_table = df_table.merge(df_plotly_country_codes, left_on="country", right_on="country")

df_table.drop(columns="gdp_(billions)", inplace=True)

#df_table.head()

df_final = df_final.groupby(["country"]).sum().sort_values("confirmed", ascending=False).drop(columns={"sno"}).head(15)

df_final.style.background_gradient(cmap='Reds')
import plotly.graph_objects as go



access_token = "pk.eyJ1IjoiZXZyZW5lcm1pczkyIiwiYSI6ImNrN3ozaHg1bzAyNGIzaW5xa2h0eHo1eGEifQ.jruBMFuoC9bh_sG6UReWQQ"

fig = go.Figure(data=go.Choropleth(

    locations=df_table['code'], # Spatial coordinates

    z = df_table['confirmed'], # Data to be color-coded

    colorscale = 'RdBu_r',

    zmax=2000,

    zmin=0,

    colorbar_title = "# Confirmed",

    )

)

    

fig.update_layout(

    hovermode='closest',

    title={

        'text': "COVID-19 Status as of 19/03/2020",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    title_font_size=15,

    autosize=False,

    width=700,

    height=450,

    mapbox=go.layout.Mapbox(

        accesstoken= access_token,

        bearing=0,

        center=go.layout.mapbox.Center(

            lat=45.124764,

            lon=3.973130

        ),

        pitch=0,

        zoom=1

    )

)



fig.show()
df_v2 = df.merge(df_plotly_country_codes, left_on="country", right_on="country")

df_v2.drop(columns="gdp_(billions)", inplace=True)

df_time = df_v2.groupby(["last_update_str","country","code"]).sum().reset_index().sort_values("last_update_str",ascending=True).drop(columns="sno")



# To balance China dominance on the map bubble graph, I have multiplied the other countries' confirmed incident number with 10 and China with 3 and saved as "Size"

# Solution has been taken from @FatihBilgin's Notebook

df_time["size"] = np.where(df_time['country']=='China', df_time['confirmed']*3, df_time['confirmed']*10) 
import plotly.express as px



fig = px.scatter_geo(df_time, locations="code", color="confirmed",

                     hover_name="country", size="size",

                     animation_frame="last_update_str",

                     projection="natural earth")



fig.update_layout(

    hovermode='closest',

    title={

        'text': "COVID-19 Status as of 19/03/2020",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    title_font_size=15,

    autosize=False,

    width=700,

    height=450,

    showlegend=False,

    mapbox=go.layout.Mapbox(

        accesstoken= access_token,

        bearing=0,

        center=go.layout.mapbox.Center(

            lat=45.124764,

            lon=3.973130

        ),

        pitch=0,

        zoom=1

    )

)



fig.show()

import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Add data



df_confirmed_table = df_confirmed.groupby("country").sum().reset_index().set_index("country")

df_death_table = df_death.groupby("country").sum().reset_index().set_index("country")

df_recovered_table = df_recovered.groupby("country").sum().reset_index().set_index("country")



x = df_confirmed.columns.tolist()



y_confirmed_china = df_confirmed_table.loc["China"]

y_death_china = df_death_table.loc["China"]

y_recovered_china = df_recovered_table.loc["China"]



y_confirmed_italy = df_confirmed_table.loc["Italy"]

y_death_italy = df_death_table.loc["Italy"]

y_recovered_italy = df_recovered_table.loc["Italy"]



fig = make_subplots(rows=1, cols=2,subplot_titles=("China", "Italy"))

                           

# Row = 1, Column = 1 

# --------------------------------------------------------------------------------------------------

fig.add_trace(go.Scatter(x=x, y=y_confirmed_china, name='China Confirmed',

                         line=dict(color='royalblue', width=4)),row =1, col = 1)



fig.add_trace(go.Scatter(x=x, y=y_death_china, name = 'China Deaths',

                         line=dict(color='firebrick', width=4)),row =1, col = 1)



fig.add_trace(go.Scatter(x=x, y=y_recovered_china, name='China Recovered',

                         line=dict(color='rgb(151,255,0)', width=4)),row =1, col = 1)



# Row = 1, Column = 2 

# --------------------------------------------------------------------------------------------------

fig.add_trace(go.Scatter(x=x, y=y_confirmed_italy, name='Italy Confirmed',

                         line=dict(color='royalblue', width=4)),row =1, col = 2)



fig.add_trace(go.Scatter(x=x, y=y_death_italy, name = 'Italy Deaths',

                         line=dict(color='firebrick', width=4)),row =1, col = 2)



fig.add_trace(go.Scatter(x=x, y=y_recovered_italy, name='Italy Recovered',

                         line=dict(color='rgb(151,255,0)', width=4)),row =1, col = 2)





# Edit the layout

fig.update_layout(title={

                        'text': "Trend of Confirmed/Recovered/Deaths ",

                        'y':0.9,

                        'x':0.5,

                        'xanchor': 'center',

                        'yanchor': 'top'},

                 title_font_size=15,

                 xaxis_title='Days',

                 legend_orientation="h",

                 yaxis_title='# Incidents',

                 legend=dict(

                         x=0.10,

                         y=-0.4,

                         traceorder="normal",

                         font=dict(

                         family="sans-serif",

                         size=9,

                         color="black"

                           ),

                        bgcolor="White",

                        bordercolor="Black",

                        borderwidth=2

                    ),

                 )



fig.show()
import plotly.graph_objects as go

from plotly.subplots import make_subplots



x = df_confirmed.columns.tolist()



y_confirmed_iran = df_confirmed_table.loc["Iran"]

y_death_iran = df_death_table.loc["Iran"]

y_recovered_iran = df_recovered_table.loc["Iran"]



y_confirmed_spain = df_confirmed_table.loc["Spain"]

y_death_spain = df_death_table.loc["Spain"]

y_recovered_spain = df_recovered_table.loc["Spain"]





fig = make_subplots(rows=1, cols=2,subplot_titles=("Iran", "Spain"))



# Create and style traces



# Row = 1, Column = 1 

fig.add_trace(go.Scatter(x=x, y=y_confirmed_iran, name='Iran Confirmed',

                         line=dict(color='royalblue', width=4)),row =1, col = 1)



fig.add_trace(go.Scatter(x=x, y=y_death_iran, name = 'Iran Deaths',

                         line=dict(color='firebrick', width=4)),row =1, col = 1)



fig.add_trace(go.Scatter(x=x, y=y_recovered_iran, name='Iran Recovered',

                         line=dict(color='rgb(151,255,0)', width=4)),row =1, col = 1)



# Row = 1, Column = 2

fig.add_trace(go.Scatter(x=x, y=y_confirmed_spain, name='Spain Confirmed',

                         line=dict(color='royalblue', width=4)),row =1, col = 2)



fig.add_trace(go.Scatter(x=x, y=y_death_spain, name = 'Spain Deaths',

                         line=dict(color='firebrick', width=4)),row =1, col = 2)



fig.add_trace(go.Scatter(x=x, y=y_recovered_spain, name='Spain Recovered',

                         line=dict(color='rgb(151,255,0)', width=4)),row =1, col = 2)



# Edit the layout

fig.update_layout(title={

                        'text': "Trend of Confirmed/Recovered/Deaths ",

                        'y':0.9,

                        'x':0.5,

                        'xanchor': 'center',

                        'yanchor': 'top'},

                 title_font_size=15,

                 xaxis_title='Days',

                 legend_orientation="h",

                 yaxis_title='# Incidents',

                 legend=dict(

                         x=0.09,

                         y=-0.4,

                         traceorder="normal",

                         font=dict(

                         family="sans-serif",

                         size=9,

                         color="black"

                           ),

                        bgcolor="White",

                        bordercolor="Black",

                        borderwidth=2

                    ),

                 )



fig.show()
df_final_subplots = df_final.drop(columns={"active","confirmed"})



import plotly.graph_objects as go

from plotly.subplots import make_subplots



colors = ['rgb(244,165,130)','rgb(146,197,222)']



labels = ["Deaths", "Recovered"]



# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=2, cols=2, specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=df_final_subplots.loc["China"], name="China"),

              1, 1)

fig.add_trace(go.Pie(labels=labels, values=df_final_subplots.loc["Italy"], name="Italy"),

              1, 2)

fig.add_trace(go.Pie(labels=labels, values=df_final_subplots.loc["Iran"], name="Iran"),

              2, 1)

fig.add_trace(go.Pie(labels=labels, values=df_final_subplots.loc["Spain"], name="Spain"),

              2, 2)

# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.35, hoverinfo="label+text+name", textinfo="value", marker=dict(colors=colors, line=dict(color='#000000', width=2)))



fig.update_layout(

    title={

        'text': "Death vs Recovered Status as of 19/03/2020",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    title_font_size=15,

    autosize=False,

    width=750,

    height=650,

    legend_orientation="h",

    legend=dict(

        x=0.35,

        y=0,

        traceorder="normal",

        font=dict(

            family="sans-serif",

            size=12,

            color="black"

        ),

        bgcolor="White",

        bordercolor="Black",

        borderwidth=2

    ),

    title_x=0.5, # Position of the title

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='China', x=0.19, y=0.82, font_size=14, showarrow=False),

                 dict(text='Italy', x=0.8, y=0.82, font_size=14, showarrow=False),

                 dict(text='Iran', x=0.20, y=0.18, font_size=14, showarrow=False),

                 dict(text='Spain', x=0.81, y=0.18, font_size=14, showarrow=False)]



)

fig.show()

import plotly.graph_objects as go





x = df_confirmed.columns.tolist()



y_confirmed_turkey = df_confirmed_table.loc["Turkey"]

y_death_turkey = df_death_table.loc["Turkey"]

y_recovered_turkey = df_recovered_table.loc["Turkey"]





fig = make_subplots(rows=1, cols=1)



# Create and style traces



# Row = 1, Column = 1 

fig.add_trace(go.Scatter(x=x, y=y_confirmed_turkey, name='Turkey Confirmed',

                         line=dict(color='royalblue', width=4)),row =1, col = 1, secondary_y=False)



fig.add_trace(go.Scatter(x=x, y=y_death_turkey, name = 'Turkey Deaths',

                         line=dict(color='firebrick', width=4)),row =1, col = 1)



fig.add_trace(go.Scatter(x=x, y=y_recovered_turkey, name='Turkey Recovered',

                         line=dict(color='rgb(151,255,0)', width=4)),row =1, col = 1)



# Edit the layout

fig.update_layout(title={

                        'text': "Trend of Confirmed/Recovered/Deaths - Turkey Only",

                        'y':0.9,

                        'x':0.5,

                        'xanchor': 'center',

                        'yanchor': 'top'},

                 title_font_size=15,

                 xaxis_title='Days',

                 legend_orientation="h",

                 yaxis_title='# Incidents',

                 legend=dict(

                         x=0.20,

                         y=-0.4,

                         traceorder="normal",

                         font=dict(

                         family="sans-serif",

                         size=9,

                         color="black"

                           ),

                        bgcolor="White",

                        bordercolor="Black",

                        borderwidth=2

                    ),

                 )



fig.show()