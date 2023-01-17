import os

os.listdir("../input/")
import pandas as pd

entities = pd.read_csv("../input/Entities.csv")

entities.head(3)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")



f, axarr = plt.subplots(2, 2, figsize=(12, 11))

f.subplots_adjust(hspace=0.75)

plt.suptitle('Cash-Stash Entity Breakdown', fontsize=18)



entities['sourceID'].value_counts().plot.bar(ax=axarr[0][0])

axarr[0][0].set_title("Data Source")



entities['service_provider'].value_counts(dropna=False).plot.bar(ax=axarr[0][1])

axarr[0][1].set_title("Money Manager")



entities['jurisdiction_description'].value_counts().head(10).plot.bar(ax=axarr[1][0])

axarr[1][0].set_title("Jurisdiction (n=10)")



entities['countries'].value_counts(dropna=False).head(10).plot.bar(ax=axarr[1][1])

axarr[1][1].set_title("Home Country (n=10)")
incorporation_dates = pd.to_datetime(entities.incorporation_date)

inactivation_dates = pd.to_datetime(entities.inactivation_date)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")



f, axarr = plt.subplots(1, 2, figsize=(12, 4))

f.subplots_adjust(hspace=0.75)



incorporation_dates.dt.year.dropna().astype(int).value_counts().head(20).sort_index().plot.bar(

    ax=axarr[0]

)

axarr[0].set_title("Year of Incorporation")



(inactivation_dates - incorporation_dates).dropna().map(lambda v: v.days).plot.hist(

    ax=axarr[1], bins=200

)

axarr[1].set_title("Fund Survival Time (Days)")

axarr[1].set_xlim([0, 15000])

pass
addresses = pd.read_csv("../input/Addresses.csv")

addresses.head(3)
addresses['countries'].value_counts().head(40).plot.bar(

    title='Addresses Mentioned by Country (n=40)', figsize=(12, 6)

)
import pandas as pd

officers = pd.read_csv("../input/Officers.csv")

officers.head(3)
officers['countries'].value_counts(dropna=False).head(40).plot.bar(

    title='Country of Origin of Officers (n=40)', figsize=(12, 6)

)
edges = pd.read_csv("../input/all_edges.csv")

edges.head(3)