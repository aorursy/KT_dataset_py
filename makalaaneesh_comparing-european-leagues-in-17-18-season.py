import pandas as pd

import json

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from collections import defaultdict


    



def combine_data_sets(country_filepath_dict):

    dfs = []

    for country, filepath in country_filepath_dict.items():

        print(filepath)

        print(country)

        df = pd.read_json(filepath)

        print(len(df))

        df['country'] = country

        dfs.append(df)

    return pd.concat(dfs)









country_filepath = {

    "england" : "../input/events_England.json",

    "spain" : "../input/events_Spain.json",

    "germany" : "../input/events_Germany.json",

    "france" : "../input/events_France.json",

    "italy": "../input/events_Italy.json"

}

combined_df = combine_data_sets(country_filepath)

def describe_event(combined_df, event):

    countries = list(country_filepath.keys())

    

    # Generating metrics

    event_country_series_map = dict()

    event_df = combined_df[combined_df['eventName'] == event]

    

    for country in countries:

        country_event_df = event_df[event_df['country'] == country]

        match_groups = country_event_df.groupby("matchId")

        event_counts_per_match = match_groups.eventId.agg('count')

        event_country_series_map[country] = event_counts_per_match

    

    fig = make_subplots(rows=1, cols=2, specs = [[dict(type="polar"), dict(type="box")]],

                       subplot_titles=["Average %ss per game in 17/18 season" %(event,),

                                      "%ss per game in 17/18 season" %(event,)])

    

    # ScatterPolar

    country_event_counts_means = [ecs.mean() for ecs in 

            [event_country_series_map[country] for country in countries]]

    fig.add_trace(go.Scatterpolar(

          r=country_event_counts_means,

          theta=countries,

          fill='toself',

          

          name=event),

                 row=1,col=1)

    

    # BOX

    for country in countries:

        fig.add_trace(go.Box(y=event_country_series_map[country], name=country),)

    

    

    event_count_max = max(country_event_counts_means)

    event_count_min = min(country_event_counts_means)

    event_count_range = event_count_max - event_count_min

    event_count_range_buffer = 0.25 * event_count_range

    fig.update_layout(

            polar=dict(radialaxis=dict(visible=True,

                                      range=[event_count_min-event_count_range_buffer,

                                            event_count_max])),

            showlegend=False

            )

    fig.show()
describe_event(combined_df, "Pass")
describe_event(combined_df, "Shot")
describe_event(combined_df, "Foul")