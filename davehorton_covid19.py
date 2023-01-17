import numpy as np 
import pandas as pd
import plotly.graph_objects as go
import os.path, time, datetime

# File processing
DataFile = "../input/uk-daily-confirmed-cases/UKDailyConfirmedCases.csv"
UKDailyConfirmedCases = pd.read_csv(DataFile, encoding = "iso-8859-1", dayfirst=True, parse_dates=['DateVal'])
UKDailyConfirmedCases[["DailyDeaths", "CumDeaths"]] = UKDailyConfirmedCases[["DailyDeaths", "CumDeaths"]].fillna(0)
UpdatedDate = time.ctime(os.path.getmtime(DataFile))

# Line chart showing new infections and cumulative infections
# UKDailyScatter = go.Figure()
# UKDailyScatter.add_trace(go.Scatter(x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["CMODateCount"], name="New Infected", line_color="crimson"))
# UKDailyScatter.add_trace(go.Scatter(x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["CumCases"], name="Total Infected", line_color="blue"))
# UKDailyScatter.add_trace(go.Scatter(x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["DailyDeaths"], name="Daily Deaths", line_color="orange"))
# UKDailyScatter.update_layout(title_text="<b>Public Health England COVID-19 UK Confirmed Daily Statistics</b> (" + UpdatedDate + ")", xaxis_rangeslider_visible=False, template="plotly_dark")
# UKDailyScatter.show()

# Now just show deaths and daily infections
UKDailyScatter = go.Figure()
UKDailyScatter.add_trace(go.Scatter(x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["CMODateCount"], name="New Cases", line_color="blue"))
UKDailyScatter.add_trace(go.Scatter(x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["DailyDeaths"], name="Daily Deaths", line_color="crimson"))
UKDailyScatter.update_layout(title_text="<b>Public Health England COVID-19 UK daily deaths and new cases</b> (" + UpdatedDate + ")<br>from 16th August deaths within 28 days of +ve test", 
    barmode="stack", xaxis_rangeslider_visible=False, template="plotly_dark",
    annotations=[
        dict(
            x="2020-03-23",
            y=967,
            xref="x",
            yref="y",
            text="Lockdown Begins",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        ),
        dict(
            x="2020-04-16",
            y=4618,
            xref="x",
            yref="y",
            text="Lockdown Extended",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        ),
        dict(
            x="2020-04-29",
            y=765,
            xref="x",
            yref="y",
            text="All settings reported",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        )
    ])
UKDailyScatter.show()

#show the 7 day moving average for the number of daily deaths
UKDailyScatter = go.Figure()
UKDailyScatter.add_trace(go.Scatter(x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["DailyDeath7DayAvg"], name="Average Deaths", line_color="crimson")),
UKDailyScatter.add_trace(go.Scatter(x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["DailyDeaths"], name="Deaths", line_color="blue"))
UKDailyScatter.update_layout(title_text="<b>7 day moving average of UK daily deaths</b> (" + UpdatedDate + ")<br>from 29th April death figures include hospital and elsewhere.", template="plotly_dark",
    annotations=[
        dict(
            x="2020-03-23",
            y=50,
            xref="x",
            yref="y",
            text="Lockdown Begins",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        ),
        dict(
            x="2020-04-16",
            y=861,
            xref="x",
            yref="y",
            text="Lockdown Extended",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        ),
        dict(
            x="2020-04-29",
            y=765,
            xref="x",
            yref="y",
            text="All settings reported",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        ),
        dict(
            x="2020-05-13",
            y=490,
            xref="x",
            yref="y",
            text="1st lockdown easing",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        ),
        dict(
            x="2020-06-01",
            y=390,
            xref="x",
            yref="y",
            text="2nd lockdown easing",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        ),
        dict(
            x="2020-08-16",
            y=190,
            xref="x",
            yref="y",
            text="Reporting deaths changes.",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        )
    ])
UKDailyScatter.show()

# Bar chart showing new infection and cumulative infections. The bar show the total cases with the stack overlayed rather than
# placed on top so that the new cases are shown as the percentage of the total cases
#UKDailyStackedBar = go.Figure(data=[
#    go.Bar(name='Existing Cases', x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["CumCases"].astype(int)-UKDailyConfirmedCases["CMODateCount"].astype(int)--UKDailyConfirmedCases["DailyDeaths"].astype(int)),
#    go.Bar(name='Daily Deaths', x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["DailyDeaths"], marker_color='crimson'),
#    go.Bar(name='New Cases', x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["CMODateCount"], marker_color='orange')
#])
#UKDailyStackedBar.update_layout(
#    title_text="<b>Public Health England COVID-19 UK Confirmed Daily Statistics</b> (" + UpdatedDate + ")",
#    barmode="stack",
#    hovermode="closest",
#    template="plotly_dark",
#    xaxis_rangeslider_visible=False)
#UKDailyStackedBar.show(1
# Bar chart showing daily cases of COVID-19 in the UK
import plotly.express as px
DailyCases = px.bar(UKDailyConfirmedCases, x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["CMODateCount"])

DailyCases.update_layout(
    title_text="<b>Public Health England COVID-19 UK Daily Cases</b> (" + UpdatedDate + ")", 
    barmode="stack",
    hovermode="closest",
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    annotations=[
        dict(
            x="2020-03-23",
            y=967,
            xref="x",
            yref="y",
            text="Lockdown Begins",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        ),
        dict(
            x="2020-04-16",
            y=4618,
            xref="x",
            yref="y",
            text="Lockdown Extended",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-100
        )
    ])
DailyCases.show()
# Daily UK cases and deaths
UKDailyOverlayStackedBar = go.Figure(data=[
    go.Bar(name='Cases', x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["CMODateCount"].astype(int)),
    go.Bar(name='Deaths', x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["DailyDeaths"], marker_color='crimson')
])
UKDailyOverlayStackedBar.update_layout(
    title_text="<b>Public Health England COVID-19 UK Daily cases and deaths</b> (" + UpdatedDate + ")", 
    barmode="group",
    hovermode="closest",
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    annotations=[
            dict(
                x="2020-03-23",
                y=967,
                xref="x",
                yref="y",
                text="Lockdown Begins",
                showarrow=True,
                align="center",
                arrowhead=7,
                arrowsize=1,
                arrowwidth=2,
                borderpad=6,
                ax=0,
                ay=-100
            ),
            dict(
                x="2020-04-16",
                y=4618,
                xref="x",
                yref="y",
                text="Lockdown Extended",
                showarrow=True,
                align="center",
                arrowhead=7,
                arrowsize=1,
                arrowwidth=2,
                borderpad=6,
                ax=0,
                ay=-100
            )
    ])
UKDailyOverlayStackedBar.show()
from plotly.subplots import make_subplots
pd.set_option('precision', 0)
GlobalDailyCases = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", encoding="iso-8859-1", parse_dates=["ObservationDate"])

# Prepare the data and deal with missing data
GlobalDailyCases["Infected"] = GlobalDailyCases["Confirmed"] - GlobalDailyCases["Deaths"] - GlobalDailyCases["Recovered"]
GlobalDailyCases.rename(columns={"Country/Region":"Country"}, inplace=True)
GlobalDailyCases["Country"] = GlobalDailyCases["Country"].replace("Mainland China", "China")
GlobalDailyCases.drop(columns="Province/State")
GlobalDailyCases[["Confirmed", "Deaths", "Recovered", "Infected"]] = GlobalDailyCases[["Confirmed", "Deaths", "Recovered", "Infected"]].fillna(0)

# Create a dataset of China and the rest of the world
Latest = GlobalDailyCases[GlobalDailyCases["ObservationDate"] == max(GlobalDailyCases["ObservationDate"])].reset_index()
#China = Latest[Latest["Country"]=="China"]
#Latest = Latest[Latest["Country"]!='China']
LatestDate = max(GlobalDailyCases["ObservationDate"]).strftime("%d/%m/%Y")

# Structure the data based on country excluding China
CountryData = Latest.groupby("Country")["Confirmed", "Deaths", "Recovered", "Infected"].sum().reset_index()
CountrySummary = CountryData.groupby("Country")["Confirmed", "Deaths", "Recovered", "Infected"].sum()
CountrySummary = CountrySummary.reset_index()
CountrySummary = CountrySummary.sort_values("Confirmed", ascending=True)

# Show top infected countries and deaths 
TopInfected = CountrySummary.sort_values("Confirmed", ascending=False).head(20).sort_values("Confirmed", ascending=True)
TopInfectedChart = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5])
TopInfectedChart.add_trace(go.Bar(
    y=TopInfected["Country"] + ' ',
    x=TopInfected["Confirmed"],
    name="Cases",
    orientation="h",
    text=TopInfected["Confirmed"],
    textposition='outside',
    marker=dict(color='rgba(56, 119, 255, 1)')
),row=1, col=1)
TopInfectedChart.add_trace(go.Bar(
    y=TopInfected["Country"] + ' ',
    x=TopInfected["Deaths"],
    name="Deaths",
    orientation="h",
    text=TopInfected["Deaths"],
    textposition='inside',
    marker=dict(color='rgba(255, 71, 71, 1)')
),row=1, col=2)
TopInfectedChart.update_layout(
    barmode="stack",
    title_text='<b>COVID-19 Top infected countries (' + LatestDate + ')</b>' ,
    template="plotly_dark",
    width=1400, 
    height=800,
    hovermode="closest"
)
TopInfectedChart.show()
# Global chart 
GlobalChart = go.Figure()
GlobalChart.add_trace(go.Bar(
    y=CountrySummary["Country"] + ' ',
    x=CountrySummary["Infected"],
    name="Infected",
    orientation="h",
    marker=dict(color='rgba(56, 119, 255, 1)')
))
GlobalChart.add_trace(go.Bar(
    y=CountrySummary["Country"] + ' ',
    x=CountrySummary["Recovered"],
    name="Recovered",
    orientation="h",
    marker=dict(color='rgba(0, 209, 91, 1)')
))
GlobalChart.add_trace(go.Bar(
    y=CountrySummary["Country"] + ' ',
    x=CountrySummary["Deaths"],
    name="Deaths",
    orientation="h",
    marker=dict(color='rgba(255, 71, 71, 1)')
))
GlobalChart.update_layout(
    barmode="stack",
    title_text='<b>COVID-19 Global daily statistics (' + LatestDate + ')</b>' ,
    template="plotly_dark",
    width=1000, 
    height=3500,
    hovermode="closest"
)
GlobalChart.show()
PopulationFile = "../input/country-population/CountryPopulation.csv"
CountryPopulation = pd.read_csv(PopulationFile, encoding="iso-8859-1")

CountryPopulation["Country"] = CountryPopulation["Country"].replace("United Kingdom", "UK")
CountryPopulation["Country"] = CountryPopulation["Country"].replace("United States", "US")


# Show population for each country
PopulationBarChart = go.Figure()
PopulationBarChart.add_trace(go.Bar(
    x=CountryPopulation["Population"],
    y=CountryPopulation["Country"] + " ",
    name="Population",
    orientation="h"
    )
)
# show confirmed covid-19 cases per country
PopulationBarChart.add_trace(go.Bar(
    y=CountrySummary["Country"] + ' ',
    x=CountrySummary["Confirmed"],
    name="Cases",
    orientation="h",
    marker=dict(color='rgba(255, 71, 71, 1)')
))
PopulationBarChart.update_layout(
    barmode="stack",
    title_text='<b>Country Population 2020</b>' ,
    template="plotly_dark",
    width=1000, 
    height=3800,
    hovermode="closest",
    yaxis=dict(autorange="reversed")
)
PopulationBarChart.show()
# Bar chart showing percentage increase in daily deaths
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        name="daily death increase",
        x=UKDailyConfirmedCases.DateVal, 
        y=UKDailyConfirmedCases["DeathPercent"]
    ))
fig.add_trace(
    go.Bar(
        name="daily cases increase",
        x=UKDailyConfirmedCases.DateVal, 
        y=UKDailyConfirmedCases["IncreasePercent"]
    ))
fig.update_layout(
    title_text="<b>Public Health England COVID-19 UK percentage increase in cumulative cases</b> (" + UpdatedDate + ")", 
    hovermode="closest",
    template="plotly_dark"
)
fig.show()

import plotly.express as px
DailyCases = px.bar(UKDailyConfirmedCases, x=UKDailyConfirmedCases.DateVal, y=UKDailyConfirmedCases["IncreasePercent"])
DailyCases.update_layout(
    title_text="<b>Public Health England COVID-19 UK percentage increase in daily deaths</b> (" + UpdatedDate + ")", 
    barmode="stack",
    hovermode="closest",
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    annotations=[
        dict(
            x="2020-03-23",
            y=25,
            xref="x",
            yref="y",
            text="Lockdown Begins",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-50
        ),
        dict(
            x="2020-04-16",
            y=15,
            xref="x",
            yref="y",
            text="Lockdown Extended",
            showarrow=True,
            align="center",
            arrowhead=7,
            arrowsize=1,
            arrowwidth=2,
            borderpad=6,
            ax=0,
            ay=-50
        )
    ])
DailyCases.show()