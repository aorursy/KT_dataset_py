# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

wlcota_df = pd.read_csv("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv")

drop_columns = ["deaths_per_100k_inhabitants", "totalCases_per_100k_inhabitants", "deathsMS", 
                "deaths_by_totalCases", "tests_per_100k_inhabitants", "totalCasesMS", "tests"]
wlcota_df = wlcota_df.drop(columns=drop_columns)
wlcota_df.dropna().head()
import yaml

# Reading states static information, such as 
# estimated population, size of the state...
with open('/kaggle/input/state_pop.yaml') as file:
    state_pop = yaml.load(file, Loader=yaml.FullLoader)

# Creating the correct component values
country_df = wlcota_df.where(wlcota_df["state"] == "TOTAL").dropna(subset=["state"]).reset_index().dropna()
active_infected = [ country_df["totalCases"].iloc[0] ]
country_df["newRecovered"] = country_df["recovered"].diff()
for nc, nd, nr in zip(country_df["newCases"].iloc[1:], 
                      country_df["newDeaths"].iloc[1:], 
                      country_df["newRecovered"].iloc[1:]):
    active_infected.append(active_infected[-1] + nc - nd - nr)
country_df["activeCases"] = active_infected

# Plotting the correct results components

fig = go.Figure()
fig.add_trace(go.Scatter(
    name="Active Cases",
    x=country_df["date"],
    y=country_df["activeCases"],
    mode='lines',
    line_shape='spline',
    line=dict(width=3)))
fig.add_trace(go.Scatter(
    name="Deaths",
    x=country_df["date"],
    y=country_df["deaths"],
    mode='lines',
    line_shape='spline'))
fig.add_trace(go.Scatter(
    name="Recovered",
    x=country_df["date"],
    y=country_df["recovered"],
    mode='lines',
    line_shape='spline'))

fig.update_layout(
    template='xgridoff',
    xaxis=dict(showgrid=False),
    xaxis_title='Date',
    legend_orientation="h", legend=dict(x=0.35, y=1.0),
    title_text="Main SIRD data components for Brazil")
iplot(fig)

# Get the state labels
states_list = wlcota_df["state"].unique().tolist()
states_list.pop(states_list.index("TOTAL"))

# Plot the recovered data at each state
fig = go.Figure()
for state in states_list:
    state_df = wlcota_df.where(wlcota_df["state"] == state).dropna()
    fig.add_trace(go.Scatter(
        name=state,
        x=state_df["date"],
        y=state_df["recovered"],
        mode='lines',
        line_shape='spline'))
    
fig.update_layout(
    template='xgridoff',
    xaxis=dict(showgrid=False),
    xaxis_title='Date',
    title_text="Recovered data collected for each state")
iplot(fig)


import numpy as np

from datetime import timedelta, datetime

# Create the interpolated dataframe for each state
state_df = pd.DataFrame()
for state in states_list:
    # Getting the state data
    new_df = wlcota_df.where(wlcota_df["state"] == state).dropna()
    # Tranform the datetime from datetime64 to datetime
    new_df["date"] = [datetime.fromisoformat(d) for d in new_df["date"].values]
    # Check if the state has enought data collected
    if len(new_df) > 5:
        # Creating the time periods for the interpolation
        first_date, last_date = new_df["date"].iloc[0], new_df["date"].iloc[-1]
        day_diff = (last_date - first_date).days
        date_vector = [first_date + timedelta(days=k) for k in range(day_diff)]
        date_df = pd.DataFrame(date_vector, columns = ['date'])
        # Merging with the datetime to interpolate
        new_df = pd.merge(date_df, new_df, how="left", on="date")
        # Interpolate (fill) the non existing data
        new_df = new_df.fillna(method="ffill", axis=0)
        # Remove the first NaN values
        new_df = new_df.dropna()
    
        # Create the correct components... Mostly here we 
        # create the active cases component...
        active_infected = [ new_df["totalCases"].iloc[0] ]
        for nc, nd, nr in zip(new_df["totalCases"].diff().iloc[1:],
                              new_df["deaths"].diff().iloc[1:],
                              new_df["recovered"].diff().iloc[1:]):
            active_infected.append(active_infected[-1] + nc - nd - nr)
        new_df["activeCases"] = active_infected 
        
        # Computing the susceptible component
        pop = state_pop[state]["population"]
        new_df["susceptible"] = pop - new_df["recovered"] - new_df["activeCases"] - new_df["deaths"]
        
        # Include this state in the general dataframe
        state_df = pd.concat((state_df, new_df.dropna()))


at_state = "RJ"
component_name = "recovered"
comp = state_df.where(state_df["state"] == at_state).dropna()
date = comp["date"].tolist()
comp = comp[component_name].tolist()

reliable_date = np.array(date)[[True] + (np.diff(comp) != 0).tolist()]
reliable_points = np.array(comp)[[True] + (np.diff(comp) != 0).tolist()]


# Show the reliable data points
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=state_df["date"].where(state_df["state"]==at_state).dropna(), 
    y=state_df[component_name].where(state_df["state"]==at_state).dropna(), 
    name="Component :{}: data".format(component_name)))
fig.add_trace(go.Scatter(
    mode='markers',
    x=reliable_date, 
    y=reliable_points, 
    name="Reliable points"))

fig.update_layout(template='xgridoff',
                  xaxis_title='Date',
                  legend_orientation="h", legend=dict(x=0.35, y=1.0),
                  title_text="Reliable points of the {} component".format(component_name))
iplot(fig)

# Getting the prediction data and historical
df = pd.read_csv("/kaggle/input/predictions.csv")

fig = go.Figure()

visibles = ["SP", "MG", "RS", "BA", "MA"]

for state in df["state"].unique():
    if state != "SC" and state != "SE" and state != "TO":
        
        s_df = df.where(df["state"]==state).dropna()
        fig.add_trace(go.Scatter3d(
            legendgroup=state,
            name=state,
            visible = True if state in visibles else "legendonly", 
            x=s_df["date"], 
            y=s_df["state"], 
            z=s_df["active_cases"]/s_df["active_cases"].max(),
            mode="lines",
            line=dict(width=6)))
        
        s_ddf = state_df.where(state_df["state"]==state).dropna()
        fig.add_trace(go.Scatter3d(
            legendgroup=state,
            showlegend=False,
            visible = True if state in visibles else "legendonly", 
            name=state,
            x=s_ddf["date"], 
            y=s_ddf["state"], 
            z=s_ddf["activeCases"]/s_df["active_cases"].max(),
            mode="markers",
            marker=dict(size=3)))
    
fig.update_layout(
    template='xgridoff',
    title_text="Active cases predictions for each state",
    xaxis_title='Date', 
    yaxis_title='States',
    scene_camera=dict(
        eye=dict(x=-1.1, y=1.0, z=0.3)
    ),
    scene=dict(
        aspectratio = dict(x=1.3,y=1.0,z=0.4),
        aspectmode = 'manual',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        zaxis=dict(showgrid=False)
    ))

iplot(fig)

fig = go.Figure()
fig.add_trace(go.Scatter(
    name="Data - Recovered",
    x=state_df["date"].where(state_df["state"] == "MA").dropna(),
    y=state_df["recovered"].where(state_df["state"] == "MA").dropna(),
    mode="markers",
    marker=dict(size=6, color="#005005")))
fig.add_trace(go.Scatter(
    name="Data - Active cases",
    x=state_df["date"].where(state_df["state"] == "MA").dropna(),
    y=state_df["activeCases"].where(state_df["state"] == "MA").dropna(),
    mode="markers",
    marker=dict(size=6, color="#38006b")))
fig.add_trace(go.Scatter(
    name="Data - Deaths",
    x=state_df["date"].where(state_df["state"] == "MA").dropna(),
    y=state_df["deaths"].where(state_df["state"] == "MA").dropna(),
    mode="markers",
    marker=dict(size=6, color="#ab000d")))

# Reading the model parameters
pars_df = pd.read_csv("/kaggle/input/state_parameters.csv")

# Computing the initial population proportion
pop_prop = pars_df["pop"].where(pars_df["state"]=="MA").dropna().values
pop = state_pop["MA"]["population"] * pop_prop

# Getting the correct model components
active_cases = df["active_cases"].where(df["state"] == "MA").dropna()
recovered = df["recovered"].where(df["state"] == "MA").dropna()
deaths = df["deaths"].where(df["state"] == "MA").dropna()
susceptible = pop - active_cases - recovered - deaths

# Getting the data vector
dates = df["date"].where(df["state"] == "MA").dropna()

fig.add_trace(go.Scatter(
    name="Model - Suceptible",
    x=dates,
    y=susceptible,
    mode="lines",
    line=dict(width=3, dash="dash", color="#0288d1")))
fig.add_trace(go.Scatter(
    name="Model - Recovered",
    x=dates,
    y=recovered,
    mode="lines",
    line=dict(width=3, dash="dash", color="#2e7d32")))
fig.add_trace(go.Scatter(
    name="Model - Active cases",
    x=dates,
    y=active_cases,
    mode="lines",
    line=dict(width=3, dash="dash", color="#6a1b9a")))
fig.add_trace(go.Scatter(
    name="Model - Deaths",
    x=dates,
    y=deaths,
    mode="lines",
    line=dict(width=3, dash="dash", color="#e53935")))


fig.update_layout(
    template='xgridoff',
    title_text="Model prediction on Maranhão Brazil",
    xaxis_title='Date')

iplot(fig)

import json

with open('/kaggle/input/brasil_estados.geojson') as handle:
    states_geo = json.load(handle)
    
pars_df = pd.read_csv("/kaggle/input/state_parameters.csv")

fig = go.Figure(go.Choroplethmapbox(geojson=states_geo, 
                                    locations=pars_df["state"], 
                                    z=pars_df["pop"],
                                    colorscale="Viridis", 
                                    marker_opacity=0.5, 
                                    marker_line_width=0, 
                                    featureidkey="properties.uf_05"))
fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=2.5, mapbox_center = {"lat": -15.7801, "lon": -47.9292})
fig.update_layout(margin={"r":0,"l":0,"b":0}, title_text="Proportion of population to attend health care systems")
iplot(fig)
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3,
                   subplot_titles=("Average days to recover", "Average Ro", "Average dailly mortality rate"))

fig.add_trace(go.Scatter(
    name="Days to recover",
    x=pars_df.sort_values(by="D")["state"],
    y=pars_df.sort_values(by="D")["D"],
    line=dict(width=3)
), row=1, col=1)

fig.add_trace(go.Scatter(
    name="One person infects...",
    x=pars_df.sort_values(by="ro")["state"],
    y=pars_df.sort_values(by="ro")["ro"],
    line=dict(width=3)
), row=1, col=2)

fig.add_trace(go.Scatter(
    name="Mortality rate",
    x=pars_df.sort_values(by="mu")["state"],
    y=pars_df.sort_values(by="mu")["mu"],
    line=dict(width=3)
), row=1, col=3)

fig.update_xaxes(title_text="States", row=1, col=1)
fig.update_xaxes(title_text="States", row=1, col=2)
fig.update_xaxes(title_text="States", row=1, col=3)

# fig.update_yaxes(title_text="Days to recover", row=1, col=1)
# fig.update_yaxes(title_text="Number of infections for each infected", row=2, col=1)
# fig.update_yaxes(title_text="Rate", row=3, col=1)

# Update title and height
fig.update_layout(title_text="Analysing the parameters information", template='xgridoff', showlegend=False)


iplot(fig)
cities_loc_df = pd.read_csv("/kaggle/input/cities_locations.csv")
cities_df = pd.read_csv("/kaggle/input/cities_start_day.csv")
cities_df = pd.merge(cities_df, cities_loc_df[["city","state","country","lat","long"]], how="left", on=["city","state"]).dropna()

fstate_df = pd.DataFrame()

for start_at in cities_df["time_from_start"].unique():
    new_df = cities_df.where(cities_df["time_from_start"] <= start_at).dropna()
    new_df["time_from_start"] = start_at
    fstate_df = pd.concat((fstate_df, new_df))

for state in cities_df["state"].unique():
    new_df = pd.DataFrame({"state": [state], "time_from_start": [0.0]})
    fstate_df = pd.concat((fstate_df, new_df))
    
fstate_df = fstate_df.sort_values(by="time_from_start")
fstate_df.head()


import plotly.express as px

fig = px.scatter_geo(fstate_df, 
                     lat="lat", 
                     lon="long",
                     color=fstate_df["state"],
                     size_max=2,
                     animation_frame="time_from_start",
                     animation_group="city",
                     scope='south america',
                     projection='natural earth')

fig.update_layout(template='xgridoff', margin={"r":0,"l":0,"b":0}, title_text="Cities contamination on time")
iplot(fig)
from geopy import distance
from geopy.point import Point

aero_df = pd.read_csv("/kaggle/input/aerodromos-publicos.csv")
aero_df["COMPRIMENTO"] = aero_df["COMPRIMENTO"].apply(lambda c: int(c[:-2]))
aero_df["ALTITUDE"] = aero_df["ALTITUDE"].apply(lambda c: float(c[:-2].replace(",",".")))
aero_df["LARGURA"] = aero_df["LARGURA"].apply(lambda c: float(c[:-2].replace(",",".")))
aero_df["NOME"] = aero_df["NOME"].apply(lambda c: c.lower())
aero_df["POSITION"] = aero_df["LATITUDE"] + " " + aero_df["LONGITUDE"]
aero_df["POSITION"] = aero_df["POSITION"].apply(lambda p: Point.from_string(p))
aero_df["LATITUDE"] = aero_df["POSITION"].apply(lambda p: p.latitude)
aero_df["LONGITUDE"] = aero_df["POSITION"].apply(lambda p: p.longitude)

# Selecting the aiport type
aero_df = aero_df.where( (aero_df["SUPERFÍCIE"] == 'Asfalto') | (aero_df["SUPERFÍCIE"] == 'Concreto') ).dropna()
aero_df = aero_df.where( aero_df["LARGURA"] >= 30.0 ).dropna()

aero_df = aero_df[["NOME", "MUNICÍPIO ATENDIDO", "UF", "LATITUDE", "LONGITUDE"]]

# Computin the ditance of nearest aiportorts

min_distance = []
cities_list, states_list = cities_df["city"].tolist(), cities_df["state"].tolist()
city_lat, city_lon = cities_df["lat"].tolist(), cities_df["long"].tolist()
aero_lat, aero_lon = aero_df["LATITUDE"].tolist(), aero_df["LONGITUDE"].tolist()
for city, state, lat, lon in zip(cities_list, states_list, city_lat, city_lon):
    
    state_aero_df = aero_df.where(aero_df["UF"] == state).dropna()
    aero_lat, aero_lon = state_aero_df["LATITUDE"].tolist(), state_aero_df["LONGITUDE"].tolist()
    aero_pos = zip(aero_lat, aero_lon)
    
    dists = [distance.distance((lat,lon),(a_lat, a_lon)).km for a_lat, a_lon in aero_pos]
    
    min_distance.append(min(dists))
    
cities_dist_df = pd.DataFrame({
    "city": cities_list, 
    "state": states_list,
    "min_dist": min_distance,
    "city_lat": city_lat,
    "city_lon": city_lon
})    

cities_dist_df["Accessibility"] = cities_dist_df["min_dist"].apply(lambda d: "by road" if d >= 40.0 else "by airport")
cities_df = pd.merge(cities_df, cities_dist_df[["city", "Accessibility", "min_dist"]], how='left', on="city")

fstate_df = pd.DataFrame()

for start_at in cities_df["time_from_start"].unique():
    new_df = cities_df.where(cities_df["time_from_start"] <= start_at).dropna()
    new_df["time_from_start"] = start_at
    fstate_df = pd.concat((fstate_df, new_df))

for state in cities_df["Accessibility"].unique():
    new_df = pd.DataFrame({"Accessibility": [state], "time_from_start": [0.0]})
    fstate_df = pd.concat((fstate_df, new_df))
    
fstate_df = fstate_df.sort_values(by="time_from_start")

fig = px.scatter_geo(fstate_df, 
                     lat="lat", 
                     lon="long",
                     color=fstate_df["Accessibility"],
                     size_max=2,
                     animation_frame="time_from_start",
                     animation_group="city",
                     scope='south america',
                     projection='natural earth')

fig.update_layout(template='xgridoff', margin={"r":0,"l":0,"b":0}, title_text="Cities contamination on time - Airport and road cities")
iplot(fig)