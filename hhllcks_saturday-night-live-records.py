import pandas as pd

import numpy as np

import bokeh

from bokeh.io import output_notebook

from bokeh.plotting import figure, show

from bokeh.models import HoverTool

output_notebook()
fs = pd.read_csv('../input/snl_season.csv', encoding="utf-8")

dfe = pd.read_csv('../input/snl_episode.csv', encoding="utf-8",parse_dates=['aired'])

dft = pd.read_csv('../input/snl_title.csv', encoding="utf-8")

dfa = pd.read_csv('../input/snl_actor.csv', encoding="utf-8")

dfat = pd.read_csv('../input/snl_actor_title.csv', encoding="utf-8")

dfr = pd.read_csv('../input/snl_rating.csv', encoding="utf-8")
dfer = pd.merge(dfe, dfr, on=['sid', 'eid'])

dfactors = pd.merge(pd.merge(dfat, dfer, on=['sid', 'eid']), dfa, on='aid')
dfactors['name'].value_counts().head(5)
df_title_season = pd.DataFrame(dfactors.groupby(['sid','name'])['aid'].count()).reset_index()

df_title_season = df_title_season.sort_values('aid', ascending=False).drop_duplicates(['sid'])

df_title_season.columns = ['Season', 'Name', 'Appearances']

df_title_season.sort_values('Season').set_index('Season')
df_host = pd.DataFrame(dfactors[dfactors.actorType == 'host'].groupby(['sid','eid','name']).count()).reset_index()

pd.DataFrame(df_host['name'].value_counts()).head(7)
df_title_cat = pd.DataFrame(dfactors.groupby(['actorType','name'])['aid'].count()).reset_index()

df_title_cat = df_title_cat.sort_values('aid', ascending=False).drop_duplicates(['actorType'])

df_title_cat.columns = ['actorType', 'Name', 'Appearances']

df_title_cat.set_index("actorType")
df_act_cat = pd.DataFrame(dfactors.groupby(['actorType','name'])['aid'].count()).reset_index()

df_act_cat.columns = ['Type', 'Name', 'Appearances']

for actorType in df_act_cat.Type.unique():

    df_act_cat[actorType] = 0

    df_act_cat.loc[df_act_cat.Type==actorType, actorType] = df_act_cat['Appearances']

    

df_act_cat = df_act_cat.drop(['Type'],axis=1)

df_act_cat = df_act_cat.groupby('Name').sum()

df_act_cat['radius'] = df_act_cat['Appearances'] / df_act_cat['Appearances'].max() * 20

#df_act_cat[df_act_cat['radius'] < 1] = 1
hover = HoverTool(

        tooltips=[

            ("Name", "@Name"),

            ("Cameos", "@cameo"),

            ("Cast", "@cast"),

            ("Crew", "@crew"),

            ("Filmed Appearance", "@filmed"),

            ("As Guest", "@guest"),

            ("As Host", "@host"),

            ("Musical Appearances", "@music"),

            ("Other", "@unknown"),

        ]

    )



TOOLS=[hover,'pan','zoom_in','zoom_out','undo','redo','reset','save','lasso_select']

p = figure(plot_width=700, plot_height=700, y_range=(-10,110),x_range=(-20,950), tools=TOOLS)

r = p.scatter("cast","host",source=df_act_cat, radius='radius')

t = show(p, notebook_handle=True)