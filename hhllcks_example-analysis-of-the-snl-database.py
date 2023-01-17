import pandas as pd

import numpy as np

import bokeh

from bokeh.io import output_notebook

from bokeh.plotting import figure, show

output_notebook()
dfs = pd.read_csv('../input/snl_season.csv', encoding="utf-8")

dfe = pd.read_csv('../input/snl_episode.csv', encoding="utf-8",parse_dates=['aired'])

dft = pd.read_csv('../input/snl_title.csv', encoding="utf-8")

dfa = pd.read_csv('../input/snl_actor.csv', encoding="utf-8")

dfat = pd.read_csv('../input/snl_actor_title.csv', encoding="utf-8")

dfr = pd.read_csv('../input/snl_rating.csv', encoding="utf-8")
dfs.head(2)
dfe.head(2)
dft.head(2)
dfa.head(2)
dfat.head(2)
dfr.head(2)
dfer = pd.merge(dfe, dfr, on=['sid', 'eid'])
dfer = dfer.sort_values(['sid', 'eid'], ascending=[True, True]).reset_index(drop=True)
# plot a trend line, too

trend = np.polyfit(dfer.index, dfer["IMDb users_avg"].values, 10)

trend_func = np.poly1d(trend)



p = figure(plot_width=800, plot_height=200, y_range=(0,10))

r = p.multi_line([dfer.index, dfer.index],[dfer["IMDb users_avg"].values, trend_func(dfer.index)], color=['blue', 'red'])

t = show(p, notebook_handle=True)
sSeasonRatingAverage = dfer.groupby("sid")["IMDb users_avg"].mean()
p = figure(plot_width=800, plot_height=200, y_range=(0,10))

r = p.line(dfer.sid.unique(),sSeasonRatingAverage.values)

t = show(p, notebook_handle=True)
dfactors = pd.merge(pd.merge(dfat, dfer, on=['sid', 'eid']), dfa, on='aid')
sActorsAppearances = dfactors.groupby('name')['sid'].count().sort_values(ascending=False)

sActorsAppearances.head(10)
dfActorsEpisodes = pd.DataFrame(dfactors.groupby(['name','sid', 'eid'])['aid'].count().sort_values(ascending=False)).reset_index()

dfActorsEpisodes.head(10)
# Define the aggregation calculations

aggregations = {

    'aid': {     # Now work on the "date" column

        'titles': 'sum',   # Find the max, call the result "max_date"

        'episodes': 'count'

    }

}

 

# Perform groupby aggregation by "month", but only on the rows that are of type "call"

dfActorsTitlePerEpisode = dfActorsEpisodes.groupby('name').agg(aggregations)

dfActorsTitlePerEpisode.columns = dfActorsTitlePerEpisode.columns.droplevel()
dfActorsTitlePerEpisode["title_avg"] = dfActorsTitlePerEpisode["titles"] / dfActorsTitlePerEpisode["episodes"]
dfActorsTitlePerEpisode[dfActorsTitlePerEpisode.episodes>=3].sort_values('title_avg', ascending=False).head(10)
dfActorsTitlePerEpisode[dfActorsTitlePerEpisode.episodes>=10].sort_values('title_avg', ascending=False).head(10)
dfActorsTitlePerEpisode[dfActorsTitlePerEpisode.episodes>=50].sort_values('title_avg', ascending=False).head(10)