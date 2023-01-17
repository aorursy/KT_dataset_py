import numpy as np

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

world = pd.read_csv("/kaggle/input/countries-of-the-world/countries of the world.csv")

world.Country = world.Country.map(lambda s: s.strip())

world = world.set_index("Country")
deaths
def normalise_by_population(row):

    try:

        return row / world.loc[row.name, "Population"] * 1000000

    except:

        return row * -1



s = deaths.loc[deaths["4/23/20"] > 40]

s = s.iloc[:,[1] + list(range(6,97))]

s = s.groupby("Country/Region").sum()

s = s.apply(normalise_by_population, axis="columns")

s.columns = pd.to_datetime(s.columns)

s = s.loc[:,"2020-03-10":].sort_values(by='4/23/20', ascending=False).iloc[:20,:].transpose()



f, ax = plt.subplots(figsize=(25, 15))

ax.set(yscale="log")

ax.set_ylim(10,1000)

sns.lineplot(data=s, ax=ax, dashes=False)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))