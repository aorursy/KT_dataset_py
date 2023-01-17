import pandas as pd
import os
import glob
import numpy as np
base_dir = '../input'
stops_df = pd.read_csv(os.path.join(base_dir, 'stops.csv'))
stops_df['stop_lon'] = stops_df['stop_lon'].map(float)
stops_df['stop_lat'] = stops_df['stop_lat'].map(float)
print(stops_df.shape[0], 'named stops')
stops_df.sample(3)
stop_times_df = pd.read_csv(os.path.join(base_dir, 'stop_times.csv'))
stop_times_df['arrival_time'] = pd.to_datetime(stop_times_df['arrival_time'], 
                                               errors='coerce')
print(stop_times_df.shape[0], 'route points')
stop_times_df.sample(3)
named_stops_df = pd.merge(stop_times_df[['trip_id','arrival_time', 'stop_sequence', 'stop_id']], 
         stops_df[['stop_id','stop_lat', 'stop_lon', 'stop_name']],
        on = 'stop_id')
named_stops_df.dropna(inplace=True)
print(named_stops_df.shape[0], 'cleaned route points')
named_stops_df.sample(3)
pts_near_zrh = named_stops_df.apply(lambda c_row: (np.square(c_row['stop_lat']-47.39)+
                     np.square(c_row['stop_lon']-8.48))<np.square(0.5), 1)
pts_near_zrh.value_counts()
if False: # keep all the points now
    named_stops_df = named_stops_df[pts_near_zrh]
print(named_stops_df.shape[0], 'points near zurich')
from collections import namedtuple
leg_class = namedtuple('leg', ['start', 'stop', 'time', 'distance'])
from tqdm import tqdm_notebook
def rows_to_legs(in_block):
    _, c_rows = in_block
    n_rows = c_rows.sort_values('stop_sequence')
    cur_trips = []
    for (_, cur_row), (_, next_row) in zip(n_rows.iterrows(), 
                                     n_rows[1:].iterrows()):
        d_time = int((next_row['arrival_time']-cur_row['arrival_time']).total_seconds())
        d_dist = int(111139*np.sqrt(np.square(next_row['stop_lat']-cur_row['stop_lat'])+
                       np.square(next_row['stop_lon']-cur_row['stop_lon'])))
        cur_trips += [leg_class(cur_row['stop_name'], next_row['stop_name'],
                       d_time,d_dist)]
    return cur_trips
        
all_stops = list(named_stops_df.groupby('trip_id'))
print(len(all_stops), 'number of stops to process')
import dask.bag as dbag
import dask.diagnostics as diag
import dask
from multiprocessing.pool import ThreadPool
from bokeh.io import output_notebook
from bokeh.resources import CDN
output_notebook(CDN, hide_banner=True)
gleg_bag = dbag.from_sequence(all_stops).map(rows_to_legs).flatten()
gleg_bag
with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof:
    with dask.config.set(pool=ThreadPool(6)):
        all_trips = gleg_bag.compute()
diag.visualize([prof, rprof])
print(len(all_trips), 'number of legs')
for _, c_leg in zip(range(5), all_trips):
    print(c_leg)
import cloudpickle
with open('all_legs.pkl', 'wb') as f:
    cloudpickle.dump(all_trips, f)
# export all trips as a csv
pd.DataFrame([x._asdict() for x in all_trips]).to_csv('all_legs.csv', index=False)
!ls -lh
