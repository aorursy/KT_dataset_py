import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

acled_keep_columns = ["event_date", "event_type", "actor1", "actor2", "fatalities", "latitude", "longitude"]
acled_csv = pd.read_csv("../input/acled-1900010120180427csv/1900-01-01-2018-04-27.csv", low_memory=False, parse_dates=["event_date"], encoding = "ISO-8859-1")
# Variables
actor = 'Al Shabaab'
date_range_start = '2017-01-01'
date_range_stop = '2018-12-31'
w = 30 # Time window in time analysis (rolling stats)
def compute_dataset(x):
    if actor in x["actor1"]:
        active_events = 1
        passive_events = 0
        killed = x['fatalities']
        losses = 0
    else:
        passive_events = 1
        active_events = 0
        killed = 0
        losses = x['fatalities']
        
    return pd.Series(data={'event_date': x['event_date'],\
                           'event_type': x['event_type'],\
                           'actor1': x['actor1'],\
                           'actor2': x['actor2'],\
                           'active_events': active_events,\
                           'killed': killed,\
                           'passive_events': passive_events,\
                           'losses': losses,\
                           'latitude': x['latitude'],\
                           'longitude': x['longitude']})

acled_data = acled_csv.filter(acled_keep_columns)\
    [((acled_csv["actor1"].str.contains(actor, na = False)) |\
      (acled_csv["actor2"].str.contains(actor, na = False)))&\
     (acled_csv['event_date'] >= date_range_start) &\
     (acled_csv['event_date'] < date_range_stop)]\
    .apply(compute_dataset, axis=1)\
    .set_index('event_date')
a = acled_data.groupby(["event_date"])['active_events', 'killed', 'passive_events', 'losses'].sum()
a.plot(subplots=True, figsize=(20, 10))
None
def compute_spread(input, killing_var):
    graph_ds = input[[killing_var, 'latitude', 'longitude']]\
        .rolling(w)\
        .aggregate({killing_var:['sum'], 'latitude': ['var', 'mean'], 'longitude': ['var', 'mean']})
    graph_ds['lat_lon_var'] = graph_ds.apply(lambda r : r['latitude']['var'] * r['longitude']['var'], axis=1)
    graph_ds.columns = [killing_var, 'lat_var', 'lat_mean', 'lon_var', 'long_mean', 'lat_lon_var']
    return graph_ds[[killing_var, 'lat_mean', 'long_mean', 'lat_lon_var']]
acled_data.plot.scatter(x='longitude', y='latitude', c='active_events', colormap='winter', figsize=(20, 10))
print("Overall map of Launched & Suffered Attacks\nGreen = Launched / Blue = Suffered")
acled_data[acled_data['active_events']==1].plot(y='latitude', style=".", figsize=(20, 10))
acled_data[acled_data['active_events']==1].plot(y='longitude', style=".", figsize=(20, 10))
print("Launched Attacks")
None
acled_data[acled_data['active_events']==0].plot(y='latitude', style=".", figsize=(20, 10))
acled_data[acled_data['active_events']==0].plot(y='longitude', style=".", figsize=(20, 10))
print("Suffered Attacks")
None
spread_active = compute_spread(input=acled_data[acled_data['active_events']==1], killing_var='killed')
spread_active.plot(subplots=True, figsize=(20, 10))
spread_active.plot.scatter(x='long_mean', y='lat_mean', c='lat_lon_var', colormap='winter', figsize=(20, 10))
print("Attacks events count: {}".format(len(spread_active)))
spread_passive = compute_spread(input=acled_data[acled_data['active_events']==0], killing_var='losses')
spread_passive.plot(subplots=True, figsize=(20, 10))
spread_passive.plot.scatter(x='long_mean', y='lat_mean', c='lat_lon_var', colormap='winter', figsize=(20, 10))
print("Suffered attacks events count: {}".format(len(spread_passive)))