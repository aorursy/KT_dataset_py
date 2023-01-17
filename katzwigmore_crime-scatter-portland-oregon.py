%matplotlib inline

import ipywidgets as widgets #buttons and value selection

import matplotlib.pyplot as plt # graphing

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
file_2015 = "../input/crime_incident_data2015.csv"

file_2016 = "../input/crime_incident_data2016.csv"

file_2017 = "../input/crime_incident_data2017.csv"
data_2015 = pd.DataFrame(pd.read_csv(file_2015)).dropna(axis=0).set_index(["Case Number"])

data_2016 = pd.DataFrame(pd.read_csv(file_2016)).dropna(axis=0).set_index(["Case Number"])

data_2017 = pd.DataFrame(pd.read_csv(file_2017)).dropna(axis=0).set_index(["Case Number"])
merged = pd.concat([data_2015, data_2016, data_2017], join="outer")

merged["Year"] = pd.to_datetime(merged["Report Month Year"]).dt.year
def plot_crimes_by_year(years=[2017,2016,2015], crimes="Larceny Offenses", density=0.008, size=1):

    plt.scatter(

        x=merged["Open Data X"].where(

          ((merged["Year"].isin(years)) & (merged["Offense Category"].isin(crimes)))

        ), 

        y=merged["Open Data Y"].where(

          ((merged["Year"].isin(years)) & (merged["Offense Category"].isin(crimes)))

        ), 

        s=size, color="b", 

        alpha=density, 

        zorder=3

    )

    plt.show()
darkness = widgets.FloatSlider(

    value=0.008,

    min=0.001,

    max=0.1,

    step=0.001,

    description='Density:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format=".001"

)
point_size = widgets.IntSlider(

    value=1,

    min=1,

    max=5,

    step=1,

    description='Point Size:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format="d"

)
year = widgets.SelectMultiple(

    options=merged["Year"].unique().tolist(),

    value=[2017, 2016, 2015],

    description='Year(s):',

    disabled=False

)
crime = widgets.SelectMultiple(

    options=merged["Offense Category"].unique().tolist(),

    value=["Larceny Offenses"],

    description='Crime(s)',

    disabled=False

)
widgets.interactive(

    plot_crimes_by_year, 

    years=year, 

    crimes=crime, 

    density=darkness, 

    size=point_size

)