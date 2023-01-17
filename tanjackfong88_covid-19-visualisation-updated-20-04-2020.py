# Importing relevant packages

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from fbprophet import Prophet

import pycountry

import plotly.express as px

import math

from scipy import stats

import seaborn as sns

from sklearn.linear_model import LinearRegression



print("Setup Complete")
# Reading the data into variables

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", 

                           parse_dates=["Last Update"])

df_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

df_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")



# Renaming columns for easier reference

df.rename(columns={"ObservationDate":"Date" , 

                    "Country/Region":"Country"}, inplace=True)

df_confirmed.rename(columns={"Country/Region":"Country"}, inplace=True)

df_deaths.rename(columns={"Country/Region":"Country"}, inplace=True)

df_recovered.rename(columns={"Country/Region":"Country"}, inplace=True)



# Reviewing the first five rows of the main dataset

print("COVID-19 Main Data")

df.head()
print("COVID-19 Death Cases Data")

df_deaths.head()
print("COVID-19 Confirmed Cases Data")

df_confirmed.head()
print("COVID-19 Recovered Cases Data")

df_recovered.head()
# Grouping the dataset

df2 = df.groupby(["Date", "Country"])[["Date", "Country", 

                                       "Confirmed", "Deaths", 

                                       "Recovered"]].sum().reset_index()



# Review the latest number of cases of COVID-19

df2.tail()
df.query("Country == 'Malaysia'").groupby("Date")[

    ["Confirmed", "Deaths", "Recovered"]].sum().reset_index()
df.query("Country == 'Singapore'").groupby("Date")[

    ["Confirmed", "Deaths", "Recovered"]].sum().reset_index()
df.query("Country == 'Mainland China'").groupby("Date")[

    ["Confirmed", "Deaths", "Recovered"]].sum().reset_index()
df.query("Country == 'Italy'").groupby("Date")[

    ["Confirmed", "Deaths", "Recovered"]].sum().reset_index()
df.query("Country == 'South Korea'").groupby("Date")[

    ["Confirmed", "Deaths", "Recovered"]].sum().reset_index()
df.query("Country == 'Spain'").groupby("Date")[

    ["Confirmed", "Deaths", "Recovered"]].sum().reset_index()
df.query("Country == 'US'").groupby("Date")[

    ["Confirmed", "Deaths", "Recovered"]].sum().reset_index()
confirmed = df.groupby("Date").sum()["Confirmed"].reset_index()

deaths = df.groupby("Date").sum()["Deaths"].reset_index()

recovered = df.groupby("Date").sum()["Recovered"].reset_index()
# Plotting the chart for worldwide cases of COVID-19

sns.set_style("darkgrid")

plt.figure(figsize=(15, 8))

plt.title("Cumulative Worldwide Cases of COVID-19 (YTD)", fontsize=25)

sns.lineplot(x="Date", y="Confirmed", data=confirmed, label="Confirmed", color="blue")

sns.lineplot(x="Date", y="Deaths", data=deaths, label="Deaths", color="red")

sns.lineplot(x="Date", y="Recovered", data=recovered, label="Recovered", color="green")

plt.xticks(rotation=90)

plt.xlabel("Date",fontsize=20)

plt.ylabel("Total no. of cases", fontsize=20)

plt.legend()
df_malaysia = df.query("Country == 'Malaysia'").groupby("Date")[

    ["Confirmed", "Deaths", "Recovered"]].sum().reset_index()



# Plotting the chart for COVID-19 cases in Malaysia

sns.set_style("darkgrid")

plt.figure(figsize=(15, 8))

plt.title("Cumulative Cases of COVID-19 in Malaysia (YTD)", fontsize=25)

sns.lineplot(x="Date", y="Confirmed", data=df_malaysia, label="Confirmed", color="blue")

sns.lineplot(x="Date", y="Deaths", data=df_malaysia, label="Deaths", color="red")

sns.lineplot(x="Date", y="Recovered", data=df_malaysia, label="Recovered", color="green")

plt.xticks(rotation=90)

plt.xlabel("Date",fontsize=20)

plt.ylabel("Total no. of cases", fontsize=20)

plt.legend()
df_confirmed = df_confirmed[["Province/State", "Lat", "Long", "Country"]]

df_latlong = pd.merge(df, df_confirmed, on=["Province/State", "Country"])

fig = px.density_mapbox(df_latlong, lat="Lat", lon="Long", 

                        hover_name="Province/State", 

                        hover_data=["Confirmed", "Deaths", "Recovered"], 

                        animation_frame="Date", color_continuous_scale="Rainbow", 

                        radius=7, zoom=0, height=700)

fig.update_layout(title="Worldwide Corona Virus Cases Time Lapse - Confirmed, Deaths, Recovered",

                  font=dict(family="Courier New, monospace",size=18,color="#7f7f7f"))

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
# Setting up worldwide growth, mortality and recovery rates of COVID-19 cases

world_df = df.groupby(["Date"]).agg({"Confirmed": ["sum"], "Recovered": ["sum"], "Deaths": ["sum"]}).reset_index()

world_df.columns = world_df.columns.get_level_values(0)

world_df["Confirmed change"] = world_df["Confirmed"].diff()

world_df["Recovered change"] = world_df["Recovered"].diff()

world_df["Deaths change"] = world_df["Deaths"].diff()

world_df["Mortality Rate"] = world_df["Deaths"] / world_df["Confirmed"]

world_df["Recovery Rate"] = world_df["Recovered"] / world_df["Confirmed"]

world_df["Growth Rate"] = (world_df["Confirmed"].diff().shift(-1) / world_df["Confirmed"]).shift(+1)

world_df["Delta Growth Rate"] = (world_df["Growth Rate"].diff().shift(-1) / world_df["Growth Rate"]).shift(+1)



# Setting up Malaysia's growth, mortality and recovery rates of COVID-19 cases

malaysia_df = df_malaysia.groupby(["Date"]).agg({"Confirmed": ["sum"], "Recovered": ["sum"], "Deaths": ["sum"]}).reset_index()

malaysia_df.columns = malaysia_df.columns.get_level_values(0)

malaysia_df["Confirmed change"] = malaysia_df["Confirmed"].diff()

malaysia_df["Recovered change"] = malaysia_df["Recovered"].diff()

malaysia_df["Deaths change"] = malaysia_df["Deaths"].diff()

malaysia_df["Mortality Rate"] = malaysia_df["Deaths"] / malaysia_df["Confirmed"]

malaysia_df["Recovery Rate"] = malaysia_df["Recovered"] / malaysia_df["Confirmed"]

malaysia_df["Growth Rate"] = (malaysia_df["Confirmed"].diff().shift(-1) / malaysia_df["Confirmed"]).shift(+1)

malaysia_df["Delta Growth Rate"] = (malaysia_df["Growth Rate"].diff().shift(-1) / malaysia_df["Growth Rate"]).shift(+1)
# Checking the latest worldwide cases

world_df.tail()
# Checking the latest cases in Malaysia

malaysia_df.tail()
fig = go.Figure()

fig.update_layout(template="seaborn")

fig.add_trace(go.Scatter(x=world_df["Date"], y=world_df["Mortality Rate"], mode="lines+markers",name="Mortality Rate", 

                         line=dict(color="red", width=2)))

fig.add_trace(go.Scatter(x=world_df["Date"], y=world_df["Recovery Rate"],mode="lines+markers", name="Recovery Rate", 

                         line=dict(color="green", width=2)))

fig.add_trace(go.Scatter(x=world_df["Date"], y=world_df["Growth Rate"], mode="lines+markers", name="Confirmed Rate", 

                         line=dict(color="blue", width=2)))

fig.update_layout(yaxis=dict(tickformat=".2%"))

fig.show()
fig = go.Figure()

fig.update_layout(template="seaborn")

fig.add_trace(go.Scatter(x=malaysia_df["Date"], y=malaysia_df["Mortality Rate"], mode="lines+markers",name="Mortality Rate", 

                         line=dict(color="red", width=2)))

fig.add_trace(go.Scatter(x=malaysia_df["Date"], y=malaysia_df["Recovery Rate"],mode="lines+markers", name="Recovery Rate", 

                         line=dict(color="green", width=2)))

fig.add_trace(go.Scatter(x=malaysia_df["Date"], y=malaysia_df["Growth Rate"], mode="lines+markers", name="Confirmed Rate", 

                         line=dict(color="blue", width=2)))

fig.update_layout(yaxis=dict(tickformat=".2%"))

fig.show()
fig = go.Figure()

fig.update_layout(template="plotly_dark")



fig.add_trace(go.Bar(x=world_df["Date"], y=world_df["Deaths change"], name="New death cases", marker_color ="red"))

fig.add_trace(go.Bar(x=world_df["Date"], y=world_df["Recovered change"], name="New recovered cases", marker_color ="green"))

fig.add_trace(go.Bar(x=world_df["Date"], y=world_df["Confirmed change"], name="New confirmed cases", marker_color ="blue"))



fig.show()
fig = go.Figure()

fig.update_layout(template="plotly_dark")



fig.add_trace(go.Bar(x=malaysia_df["Date"], y=malaysia_df["Deaths change"], name="New death cases", marker_color ="red"))

fig.add_trace(go.Bar(x=malaysia_df["Date"], y=malaysia_df["Recovered change"], name="New recovered cases", marker_color ="green"))

fig.add_trace(go.Bar(x=malaysia_df["Date"], y=malaysia_df["Confirmed change"], name="New confirmed cases", marker_color ="blue"))



fig.show()
confirmed_latest = df2.groupby(["Date", "Country"]).sum()["Confirmed"].reset_index()

deaths_latest = df2.groupby(["Date", "Country"]).sum()["Deaths"].reset_index()

recovered_latest = df2.groupby(["Date", "Country"]).sum()["Recovered"].reset_index()

latest_date = confirmed_latest["Date"].max()

print("Latest date :", latest_date)
confirmed_latest = confirmed_latest[(confirmed_latest["Date"]==latest_date)][["Country", "Confirmed"]]

deaths_latest = deaths_latest[(deaths_latest["Date"]==latest_date)][["Country", "Deaths"]]

recovered_latest = recovered_latest[(recovered_latest["Date"]==latest_date)][["Country", "Recovered"]]
confirmed_latest.sort_values(by=["Confirmed"]).tail(10).reset_index()
deaths_latest.sort_values(by=["Deaths"]).tail(10).reset_index()
confirmed_latest.sort_values(by=["Confirmed"]).head(10).reset_index()
confirmed_by_country_df = df.groupby(["Date", "Country"]).sum().reset_index()

fig = px.line(confirmed_by_country_df, x="Date", y="Confirmed", color="Country", line_group="Country", hover_name="Country")

fig.update_layout(template="plotly_dark",

yaxis_type="log")

fig.show()
# Plotting scatter plot with regression line between worldwide confirmed and death cases of COVID-19

slope, intercept, r_value, p_value, std_err = stats.linregress(df["Confirmed"], df["Deaths"])

plt.title("Relationship between Worldwide Confirmed Cases and Death Cases of COVID-19")

ax = sns.regplot(x="Confirmed", y="Deaths", data=df, 

                line_kws={"label":"y={0:.4f}x+{1:.4f}".format(slope, intercept)})

ax.legend()

plt.show()

print("R-squared: %f" % r_value ** 2)



# For every newly 1000 confirmed COVID-19 patients Worldwide

x_world = 1000

y_world_deaths = (slope * x_world) + intercept

mortality_world = (y_world_deaths / x_world) * 100

print("Estimated No. of deaths for every 1,000 confirmed COVID-19 cases: ", 

      round(y_world_deaths, 0))

print("Global mortality rate of COVID-19: ", mortality_world, "%")
# Plotting scatter plot with regression line between confirmed and death cases of COVID-19 in Malaysia

slope, intercept, r_value, p_value, std_err = stats.linregress(df_malaysia["Confirmed"], df_malaysia["Deaths"])

plt.title("Relationship between Confirmed Cases and Death Cases of COVID-19 in Malaysia")

ax = sns.regplot(x="Confirmed", y="Deaths", data=df_malaysia, 

                line_kws={"label":"y={0:.4f}x+{1:.4f}".format(slope, intercept)})

ax.legend()

plt.show()

print("R-squared: %f" % r_value ** 2)



# For every newly 1000 confirmed COVID-19 patients in Malaysia

x_malaysia = 1000

y_malaysia_deaths = (slope * x_malaysia) + intercept

mortality_malaysia = (y_malaysia_deaths / x_malaysia) * 100

print("Estimated No. of deaths for every 1,000 confirmed COVID-19 cases: ", 

      round(y_malaysia_deaths, 0))

print("Malaysia's mortality rate of COVID-19: ", mortality_malaysia, "%")