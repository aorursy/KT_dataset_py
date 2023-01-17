import pandas as pd

import numpy as np

import bokeh

from bokeh.io import output_notebook

from bokeh.plotting import figure, show

from bokeh.models import HoverTool

from bokeh.palettes import Spectral11

output_notebook()
fs = pd.read_csv('../input/snl_season.csv', encoding="utf-8")

dfe = pd.read_csv('../input/snl_episode.csv', encoding="utf-8",parse_dates=['aired'])

dft = pd.read_csv('../input/snl_title.csv', encoding="utf-8")

dfa = pd.read_csv('../input/snl_actor.csv', encoding="utf-8")

dfat = pd.read_csv('../input/snl_actor_title.csv', encoding="utf-8")

dfr = pd.read_csv('../input/snl_rating.csv', encoding="utf-8")
dfer = pd.merge(dfe, dfr, on=['sid', 'eid'])

dfactors = pd.merge(pd.merge(dfat, dfer, on=['sid', 'eid']), dfa, on='aid')
df_sea_cat = pd.DataFrame(dfactors.groupby(['actorType','sid'])['aid'].count()).reset_index()

df_sea_cat.columns = ['Type', 'Season', 'Appearances']

for actorType in df_sea_cat.Type.unique():

    df_sea_cat[actorType] = 0

    df_sea_cat.loc[df_sea_cat.Type==actorType, actorType] = df_sea_cat['Appearances']

    

df_sea_cat = df_sea_cat.drop(['Type'],axis=1)

df_sea_cat = df_sea_cat.groupby('Season').sum()
TOOLS=['pan','zoom_in','zoom_out','undo','redo','reset','save']

p = figure(plot_width=600, plot_height=600, y_range=(-10,df_sea_cat['Appearances'].max()),x_range=(0,45), tools=TOOLS)

#r = p.multi_line(['Season','Season','Season', 'Season', 'Season','Season','Season', 'Season', 'Season'],

 #                ['Appearances','cameo','cast','crew','filmed','guest','host','music','unknown'],

#              line_width=4, source=df_sea_cat)

numlines=len(df_sea_cat.columns)

mypalette=Spectral11[0:numlines]



for column in df_sea_cat.columns:

    p.line(df_sea_cat.index,df_sea_cat[column], legend=column, line_color=Spectral11[df_sea_cat.columns.get_loc(column)], line_width=3)
t = show(p, notebook_handle=True)
df_sea_cat_avg = df_sea_cat.copy()

df_sea_cat_avg['Episodes'] = dfe.groupby(['sid'])['eid'].count()
for column in df_sea_cat_avg.columns:

    if column != 'Episodes':

        df_sea_cat_avg[column] = df_sea_cat_avg[column] / df_sea_cat_avg['Episodes']
TOOLS=['pan','zoom_in','zoom_out','undo','redo','reset','save']

p = figure(plot_width=600, plot_height=600, y_range=(-10,df_sea_cat_avg['Appearances'].max()),x_range=(0,45), tools=TOOLS)



for column in df_sea_cat_avg.columns:

    p.line(df_sea_cat_avg.index,df_sea_cat_avg[column], legend=column, line_color=Spectral11[df_sea_cat_avg.columns.get_loc(column)], line_width=3)
t = show(p, notebook_handle=True)