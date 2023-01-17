import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

import folium

from folium.plugins import HeatMap, HeatMapWithTime

%matplotlib inline
birds_df = pd.read_csv("/kaggle/input/xenocanto-birds-from-india/birds_india.csv")
birds_df.head()
birds_df.info()
birds_df.describe()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(birds_df)
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return(np.transpose(tt))
unique_values(birds_df)
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()    
plot_count("sp", "Species", birds_df, size=4)
plot_count("en", "Species (English name)", birds_df, size=4)
plot_count("gen", "Latin name (for gen)", birds_df, size=4)
plot_count("rec", "Recorder person", birds_df, size=4)
plot_count("bird-seen", "`was the bird seen?`", birds_df, size=2)
plot_count("playback-used", "`was playback used?`", birds_df, size=2)
aggregated_df = birds_df.groupby(["lat", "lng"])["id"].count().reset_index()

aggregated_df.columns = ['lat', 'lng', 'count']
m = folium.Map(location=[20, 78], zoom_start=5)

max_val = max(aggregated_df['count'])

HeatMap(data=aggregated_df[['lat', 'lng', 'count']],\

        radius=15, max_zoom=12).add_to(m)

m
subset = birds_df.loc[birds_df["gen"]=="Phylloscopus"]

aggregated_df = subset.groupby(["lat", "lng"])["id"].count().reset_index()

aggregated_df.columns = ['lat', 'lng', 'count']

m = folium.Map(location=[20, 78], zoom_start=5)

max_val = max(aggregated_df['count'])

HeatMap(data=aggregated_df[['lat', 'lng', 'count']],\

        radius=15, max_zoom=12).add_to(m)

m
subset = birds_df.loc[birds_df["gen"]=="Mystery"]

aggregated_df = subset.groupby(["lat", "lng"])["id"].count().reset_index()

aggregated_df.columns = ['lat', 'lng', 'count']

m = folium.Map(location=[20, 78], zoom_start=5)

max_val = max(aggregated_df['count'])

HeatMap(data=aggregated_df[['lat', 'lng', 'count']],\

        radius=15, max_zoom=12).add_to(m)

m