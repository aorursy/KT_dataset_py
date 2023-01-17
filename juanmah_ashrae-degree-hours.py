# !pip install nb_black

# %load_ext nb_black

import numpy as np

import pandas as pd

import colorlover as cl

import plotly.express as px

import plotly.io as pio

import seaborn as sns

from IPython.core.display import display, HTML

import pickle

from tqdm.auto import tqdm
pio.templates.default = "plotly_white"

category_orders = {"meter": ["electricity", "chilledwater", "steam", "hotwater"]}

color_discrete_map_meter = {

    "electricity": "#7844c5",

    "chilledwater": "#83c9e9",

    "steam": "#dfdeeb",

    "hotwater": "#f2957a",

}
color_discrete_map_site = {

    # Europe/London -> Blues

    "1 London": cl.scales["3"]["seq"]["Blues"][2],  # London, Europe/London

    "5 Leicester": cl.scales["3"]["seq"]["Blues"][1],  # Leicester, Europe/London

    # Europe/Dublin -> Greens

    "12 Dublin": cl.scales["3"]["seq"]["Greens"][2],  # Dublin

    # Canada/Eastern -> Reds

    "7 Montréal": cl.scales["3"]["seq"]["Reds"][2],  # Montréal, Canada/Eastern

    "11 Montréal": cl.scales["3"]["seq"]["Reds"][1],  # Montréal, Canada/Eastern

    # US/Eastern -> Purples

    "0 Jacksonville": cl.scales["6"]["seq"]["Purples"][5],  # Jacksonville, US/Eastern

    "3 Philadelphia": cl.scales["6"]["seq"]["Purples"][4],  # Philadelphia, US/Eastern

    "6 Philadelphia": cl.scales["6"]["seq"]["Purples"][3],  # Philadelphia, US/Eastern

    "8 Jacksonville": cl.scales["6"]["seq"]["Purples"][2],  # Jacksonville, US/Eastern

    "14 Philadelphia": cl.scales["6"]["seq"]["Purples"][1],  # Philadelphia, US/Eastern

    "15 Pittsburgh": cl.scales["6"]["seq"]["Purples"][0],  # Pittsburgh, US/Eastern

    # US/Central -> Greys

    "9 San Antonio": cl.scales["3"]["seq"]["Greys"][2],  # San Antonio, US/Central

    "13 Minneapolis": cl.scales["3"]["seq"]["Greys"][1],  # Minneapolis, US/Central

    # US/Arizona -> Yellows

    "2 Phoenix": cl.scales["3"]["seq"]["YlOrBr"][0],  # Phoenix, US/Arizona

    # US/Pacific -> Oranges

    "4 San Francisco": cl.scales["3"]["seq"]["Oranges"][2],  # San Francisco, US/Pacific

    "10 Las Vegas": cl.scales["3"]["seq"]["Oranges"][1],  # Las Vegas, US/Pacific

}
with open("../input/ashrae-data-wrangling-train/train.pickle", "rb") as f:

    train = pickle.load(f)

with open("../input/ashrae-data-wrangling-train/timeseries.pickle", "rb") as f:

    timeseries = pickle.load(f)

with open(

    "../input/ashrae-data-wrangling-building-metadata/building_metadata.pickle", "rb"

) as f:

    building_metadata = pickle.load(f)

with open("../input/ashrae-data-wrangling-weather/weather_train.pickle", "rb") as f:

    weather_train = pickle.load(f)
weather_train["site"] = weather_train["site_id"]

weather_train["site"].cat.rename_categories(

    {

        "0": "0 Jacksonville",

        "1": "1 London",

        "2": "2 Phoenix",

        "3": "3 Philadelphia",

        "4": "4 San Francisco",

        "5": "5 Leicester",

        "6": "6 Philadelphia",

        "7": "7 Montréal",

        "8": "8 Jacksonville",

        "9": "9 San Antonio",

        "10": "10 Las Vegas",

        "11": "11 Montréal",

        "12": "12 Dublin",

        "13": "13 Minneapolis",

        "14": "14 Philadelphia",

        "15": "15 Pittsburgh",

    },

    inplace=True,

)
weather_train["month"] = weather_train["timestamp"].dt.month



weather_train_month_columns = ["month", "site",] + [

    dh for dh in weather_train.columns.to_list() if "degree_hours" in dh

]



weather_train_month = (

    weather_train[weather_train_month_columns].groupby(["month", "site"]).agg([np.sum])

)

weather_train_month.reset_index(inplace=True)

weather_train_month = weather_train_month.droplevel(1, axis="columns")

weather_train_month
site_hdh = weather_train_month.groupby("site").agg("sum")

site_hdh_list = (

    site_hdh.sort_values(by="heating_degree_hours_15", ascending=True)

    .reset_index()["site"]

    .to_list()

)

fig = px.area(

    weather_train_month,

    x="month",

    y="heating_degree_hours_15",

    category_orders={"site": site_hdh_list},

    color="site",

    color_discrete_map=color_discrete_map_site,

)

fig.update_layout()

fig.update_xaxes(dtick=1)

fig.show()
site_cdh = weather_train_month.groupby("site").agg("sum")

site_cdh_list = (

    site_cdh.sort_values(by="cooling_degree_hours_25", ascending=True)

    .reset_index()["site"]

    .to_list()

)

fig = px.area(

    weather_train_month,

    x="month",

    y="cooling_degree_hours_25",

    category_orders={"site": site_cdh_list},

    color="site",

    color_discrete_map=color_discrete_map_site,

)

fig.update_layout()

fig.update_xaxes(dtick=1)

fig.show()
degree_hours = ["air_temperature"] + [

    dh for dh in weather_train.columns.to_list() if "degree_hours" in dh

]



for degree_hour in degree_hours:

    timeseries[degree_hour] = np.NaN
def get_corr(index, building_id, meter):

    ts = train[(train["building_id"] == building_id) & (train["meter"] == meter)][

        "meter_reading"

    ]

    site = building_metadata[building_metadata["building_id"] == building_id][

        "site_id"

    ].values[0]

    weather_train_site = weather_train[weather_train["site_id"] == site]

    for degree_hour in degree_hours:

        dh = weather_train_site[degree_hour]

        #         print(index, building_id, meter, degree_hour, len(ts), len(dh))

        timeseries.loc[index, degree_hour] = np.ma.corrcoef(

            np.ma.masked_invalid(ts), np.ma.masked_invalid(dh)

        )[0, 1]





#         print(np.ma.corrcoef(np.ma.masked_invalid(ts), np.ma.masked_invalid(dh)))





# get_corr(0, "1", "electricity")

# timeseries
for index, row in tqdm(timeseries.iterrows(), total=timeseries.shape[0]):

    get_corr(index, row["building_id"], row["meter"])
timeseries_heating = timeseries[

    ["building_id", "meter", 'air_temperature']

    + [hdh for hdh in weather_train.columns.to_list() if "heating_degree_hours" in hdh]

]

timeseries_heating = pd.wide_to_long(

    timeseries_heating,

    stubnames="heating_degree_hours_",

    i=["building_id", "meter", 'air_temperature'],

    j="base_temperature",

)

timeseries_heating.reset_index(inplace=True)
fig = px.scatter(

    timeseries_heating,

    x="air_temperature",

    y="heating_degree_hours_",

    color="meter",

    animation_frame="base_temperature",    

    opacity=0.7,

    marginal_x="violin",

    marginal_y="violin",

    hover_name="building_id",

    category_orders=category_orders,

    color_discrete_map=color_discrete_map_meter,

)

fig.show()
timeseries_cooling = timeseries[

    ["building_id", "meter", 'air_temperature']

    + [cdh for cdh in weather_train.columns.to_list() if "cooling_degree_hours" in cdh]

]

timeseries_cooling = pd.wide_to_long(

    timeseries_cooling,

    stubnames="cooling_degree_hours_",

    i=["building_id", "meter", 'air_temperature'],

    j="base_temperature",

)

timeseries_cooling.reset_index(inplace=True)
fig = px.scatter(

    timeseries_cooling,

    x="air_temperature",

    y="cooling_degree_hours_",

    color="meter",

    animation_frame="base_temperature",        

    opacity=0.7,

    marginal_x="violin",

    marginal_y="violin",

    hover_name="building_id",

    category_orders=category_orders,

    color_discrete_map=color_discrete_map_meter,

)

fig.update_layout(yaxis=dict(range=[-1, 1]))

fig.show()
air_temperature_mean = (

    timeseries[["building_id", "meter", "air_temperature"]]

    .groupby("meter")

    .agg(np.mean)

)

air_temperature_mean
fig = px.histogram(

    timeseries_heating,

    x="heating_degree_hours_",

    color="meter",

    facet_col="meter",

    animation_frame="base_temperature",

    nbins=100,

    category_orders=category_orders,

    color_discrete_map=color_discrete_map_meter,

)

fig.show()
import plotly.graph_objects as go



heating_mean = (

    timeseries_heating.groupby(["meter", "base_temperature"]).agg(np.mean).reset_index()

)

fig = px.line(

    heating_mean,

    x="base_temperature",

    y="heating_degree_hours_",

    color="meter",

    category_orders=category_orders,

    color_discrete_map=color_discrete_map_meter,

)

fig.update_layout(

    shapes=[

        go.layout.Shape(

            type="line",

            x0=timeseries_heating["base_temperature"].min(),

            y0=-air_temperature_mean.loc["electricity"][0],

            x1=timeseries_heating["base_temperature"].max(),

            y1=-air_temperature_mean.loc["electricity"][0],

            line=dict(

                color=color_discrete_map_meter["electricity"], dash="dot", width=1

            ),

        ),

        go.layout.Shape(

            type="line",

            x0=timeseries_heating["base_temperature"].min(),

            y0=-air_temperature_mean.loc["chilledwater"][0],

            x1=timeseries_heating["base_temperature"].max(),

            y1=-air_temperature_mean.loc["chilledwater"][0],

            line=dict(

                color=color_discrete_map_meter["chilledwater"], dash="dot", width=1

            ),

        ),

        go.layout.Shape(

            type="line",

            x0=timeseries_heating["base_temperature"].min(),

            y0=-air_temperature_mean.loc["steam"][0],

            x1=timeseries_heating["base_temperature"].max(),

            y1=-air_temperature_mean.loc["steam"][0],

            line=dict(color=color_discrete_map_meter["steam"], dash="dot", width=1),

        ),

        go.layout.Shape(

            type="line",

            x0=timeseries_heating["base_temperature"].min(),

            y0=-air_temperature_mean.loc["hotwater"][0],

            x1=timeseries_heating["base_temperature"].max(),

            y1=-air_temperature_mean.loc["hotwater"][0],

            line=dict(color=color_discrete_map_meter["hotwater"], dash="dot", width=1),

        ),

    ]

)

fig.show()
fig = px.histogram(

    timeseries_cooling,

    x="cooling_degree_hours_",

    color="meter",

    facet_col="meter",

    animation_frame="base_temperature",

    nbins=100,

    category_orders=category_orders,

    color_discrete_map=color_discrete_map_meter,

)

fig.show()
cooling_mean = (

    timeseries_cooling.groupby(["meter", "base_temperature"]).agg(np.mean).reset_index()

)

fig = px.line(

    cooling_mean,

    x="base_temperature",

    y="cooling_degree_hours_",

    color="meter",

    category_orders=category_orders,

    color_discrete_map=color_discrete_map_meter,

)

fig.update_layout(

    shapes=[

        go.layout.Shape(

            type="line",

            x0=timeseries_cooling["base_temperature"].min(),

            y0=air_temperature_mean.loc["electricity"][0],

            x1=timeseries_cooling["base_temperature"].max(),

            y1=air_temperature_mean.loc["electricity"][0],

            line=dict(

                color=color_discrete_map_meter["electricity"], dash="dot", width=1

            ),

        ),

        go.layout.Shape(

            type="line",

            x0=timeseries_cooling["base_temperature"].min(),

            y0=air_temperature_mean.loc["chilledwater"][0],

            x1=timeseries_cooling["base_temperature"].max(),

            y1=air_temperature_mean.loc["chilledwater"][0],

            line=dict(

                color=color_discrete_map_meter["chilledwater"], dash="dot", width=1

            ),

        ),

        go.layout.Shape(

            type="line",

            x0=timeseries_cooling["base_temperature"].min(),

            y0=air_temperature_mean.loc["steam"][0],

            x1=timeseries_cooling["base_temperature"].max(),

            y1=air_temperature_mean.loc["steam"][0],

            line=dict(color=color_discrete_map_meter["steam"], dash="dot", width=1),

        ),

        go.layout.Shape(

            type="line",

            x0=timeseries_cooling["base_temperature"].min(),

            y0=air_temperature_mean.loc["hotwater"][0],

            x1=timeseries_cooling["base_temperature"].max(),

            y1=air_temperature_mean.loc["hotwater"][0],

            line=dict(color=color_discrete_map_meter["hotwater"], dash="dot", width=1),

        ),

    ]

)

fig.show()