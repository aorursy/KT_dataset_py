!pip install nb_black -q
%load_ext nb_black
import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from fbprophet import Prophet



data = pd.concat(

    [

        pd.read_csv(

            "../input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv",

            error_bad_lines=False,

        ),

        pd.read_csv(

            "../input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv",

            error_bad_lines=False,

        ),

        pd.read_csv(

            "../input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",

            error_bad_lines=False,

        ),

    ]

)

data.dtypes
data.head()
data.isna().sum()
data.drop(

    [

        "Unnamed: 0",

        "ID",

        "IUCR",

        "Beat",

        "Case Number",

        "District",

        "Ward",

        "Updated On",

        "Year",

        "Community Area",

        "X Coordinate",

        "Y Coordinate",

        "Latitude",

        "Longitude",

        "Location",

        "FBI Code",

    ],

    axis=1,

    inplace=True,

)
data.Date = pd.to_datetime(data.Date, format="%m/%d/%Y %I:%M:%S %p")

data.set_index("Date", inplace=True)
import plotly.express as px





def plot_counts(serie, title):

    df = pd.DataFrame(serie.value_counts()[:15])

    df.columns = ["Freq"]

    df["Type"] = df.index

    fig = px.bar(df, y="Freq", x="Type", text="Freq", color="Freq")

    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")

    fig.update_layout(title_text=title)

    fig.show()





plot_counts(data["Primary Type"], "Kind of Crimes")
plot_counts(data["Location Description"], "Location of Crimes")
aux = pd.DataFrame(data.resample("M").size(), columns=["Number of cases"])

aux["Month"] = aux.index.month.astype(str)

aux["Year"] = aux.index.year.astype(str)

aux["Year-Month"] = aux["Year"].str.cat(aux["Month"].str.zfill(2), sep="-")

aux.head()
aux_csm = aux[["Month", "Year", "Number of cases"]]

plt.figure(figsize=(25, 10))

sns.heatmap(

    aux_csm.pivot("Month", "Year", "Number of cases").fillna(0).astype(int),

    annot=True,

    fmt="d",

    linewidths=0.5,

    cmap="Reds",

)
fig = px.bar(

    aux,

    x="Year-Month",

    y="Number of cases",

    hover_data=["Year", "Month", "Number of cases"],

    color="Number of cases",

    text="Number of cases",

    height=600,

    width=2400,

)

fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

fig.update_layout(title_text='Crime counts per year-month in bars')

fig.show()
fig = px.bar(

    aux,

    x="Month",

    y="Number of cases",

    hover_data=["Year", "Month", "Number of cases"],

    color="Number of cases",

    text="Number of cases",

    height=600,

    width=1500,

)

fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

fig.update_layout(title_text="Crime counts per year-month in stacked bars (Month)")

fig.show()
fig = px.bar(

    aux,

    x="Year",

    y="Number of cases",

    hover_data=["Year", "Month", "Number of cases"],

    color="Number of cases",

    text="Number of cases",

    height=600,

    width=1500,

)

fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

fig.update_layout(title_text="Crime counts per year-month in stacked bars (Years)")



fig.show()
data_model = data.resample("M").size().reset_index()

data_model.columns = ["ds", "y"]

data_model
model = Prophet()

model.fit(data_model)
future = model.make_future_dataframe(periods=300)

forecast = model.predict(future)

forecast.head()
import plotly.graph_objects as go



# Create random data with numpy



# Create traces

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predict Values"

    )

)

fig.add_trace(

    go.Scatter(x=forecast["ds"], y=forecast["trend"], mode="lines", name="Trend")

)

fig.add_trace(

    go.Scatter(

        x=data_model["ds"], y=data_model["y"], mode="lines+markers", name="Real Values",

    )

)

fig.update_layout(

    title_text="Comperating the real x predicted",

    yaxis_title="Crime counts",

    xaxis_title="Date",

)





fig.show()
model.plot_components(forecast)