# !pip install nb_black

# %load_ext nb_black

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.io as pio

from IPython.core.display import display, HTML

import pickle

import itertools

from tqdm.auto import trange

pio.templates.default = "plotly_white"
with open("../input/ashrae-data-wrangling-train/train.pickle", "rb") as f:

    train = pickle.load(f)
correlation = pd.DataFrame(columns={"building_id", "energy_aspects", "correlation"})



for building_id in trange(0, 1449):

    energy = train[train["building_id"] == str(building_id)]

    energy_aspects = energy["meter"].unique().to_list()

    if len(energy_aspects) > 1:

        for combination in itertools.combinations(energy_aspects, 2):

            ea0 = energy[energy["meter"] == combination[0]]["meter_reading"]

            ea1 = energy[energy["meter"] == combination[1]]["meter_reading"]

            corr = np.ma.corrcoef(np.ma.masked_invalid(ea0), np.ma.masked_invalid(ea1))[

                0, 1

            ]

            correlation = correlation.append(

                {

                    "building_id": building_id,

                    "energy_aspects": "-".join(combination),

                    "correlation": corr,

                },

                ignore_index=True,

            )



correlation.to_csv('correlation.csv')

display(correlation)

display(HTML(f'There are {correlation.shape[0]} combinations between energy aspects for the buildings with more than one energy aspect.'))
fig = px.histogram(

    correlation,

    x="correlation",

    color="energy_aspects",

    facet_row="energy_aspects",

    nbins=50,

    height=1200,

)

fig.update_layout(showlegend=False, xaxis=dict(range=[-1, 1], dtick = 0.2))

for i in range(0, 6):

    fig.layout.annotations[i].text = fig.layout.annotations[i].text.replace(

        "energy_aspects=", ""

    )

fig.show()
correlation[abs(correlation["correlation"]) > 0.80].sort_values(

    by="correlation", ascending=False

)