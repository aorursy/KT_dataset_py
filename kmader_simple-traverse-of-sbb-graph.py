%matplotlib inline
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
base_dir = '../input/swiss-rail-plan/'
stops_df = pd.read_csv(os.path.join(base_dir, 'stops.csv'))
stops_df['stop_lon'] = stops_df['stop_lon'].map(float)
stops_df['stop_lat'] = stops_df['stop_lat'].map(float)
stops_df.sample(3)
stops_df.plot.scatter(x='stop_lon', y = 'stop_lat')
# limit range to swiss-ish places (other places skew the map too much)
stops_df['stop_lon'] = np.clip(stops_df['stop_lon'], 6, 10)
stops_df['stop_lat'] = np.clip(stops_df['stop_lat'], 45.5, 48)
fig = plt.figure(figsize=(14,10))
ax1 = fig.add_subplot(111)
west, south, east, north = 5, 45, 11, 48
m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,
            llcrnrlon=west, urcrnrlon=east, lat_ts=south, ax=ax1)
m.etopo()
m.drawcountries()
m.drawrivers()
t_lon, t_lat = m(stops_df['stop_lon'].values, stops_df['stop_lat'].values)
ax1.scatter(t_lon, t_lat)
stop_dict = {c_row['stop_name']: (c_row['stop_lon'], c_row['stop_lat'])
                                        for _, c_row in stops_df.iterrows()}
def show_stop(stop_name, ax, **kwargs):
    c_lon, c_lat = stop_dict[stop_name]
    ax.plot(c_lon, c_lat, '.', **kwargs)
    ax.text(c_lon, c_lat, stop_name)
def show_leg(start_name, end_name, ax, **kwargs):
    s_lon, s_lat = stop_dict[start_name]
    e_lon, e_lat = stop_dict[end_name]
    ax.plot([s_lon, e_lon], [s_lat, e_lat], '-')
fig, (ax1) = plt.subplots(1, 1, figsize = (10, 5))
ax1.axis('off')
for _, c_row in stops_df.sample(50).iterrows():    
    show_stop(c_row['stop_name'], ax=ax1)
import cloudpickle
with open('../input/parsing-sbb-routes-as-a-graph/all_legs.pkl', 'rb') as f:
    all_legs = cloudpickle.load(f)
print(len(all_legs), 'number of legs')
for _, c_leg in zip(range(3), all_legs):
    print(c_leg)    
fig, (ax1) = plt.subplots(1, 1, figsize = (10, 5))
unique_dict = {tuple(sorted([c_leg.start, c_leg.stop])): c_leg for c_leg in all_legs}
unique_legs = list(unique_dict.values())
print(len(unique_legs), 'number of legs')
for i, c_leg in zip(range(50), unique_legs):
    if i<=5: 
        print(c_leg)
    show_leg(c_leg.start, c_leg.stop, ax = ax1)
leg_df = pd.DataFrame(unique_legs) 
pd.melt(leg_df, value_vars=['start', 'stop'])\
    .groupby('value').size() \
    .reset_index(name='counts')\
    .sort_values('counts', ascending=False)\
    .head(5)
leg_df[leg_df['start'].isin(['Zürich HB']) | leg_df['stop'].isin(['Zürich HB'])].head(5)
def tree_calc_total_distance(start, 
                            stop, 
                            leg_list, 
                            mode = 'bfs',
                             maximum_distance = None,
                            _callback = None):
    visited_legs = set([start])
    start_stack = [[0, start, None]]
    while len(start_stack)>0:
        if mode=='bfs':
            cur_distance, cur_start, last_leg = start_stack.pop(0) 
        elif mode=='dfs':
            cur_distance, cur_start, last_leg = start_stack.pop()
        else:
            raise ValueError('Dont know how to search trees like that, {}'.format(mode))
        if (_callback is not None) and (last_leg is not None):
            _callback(last_leg.start, last_leg.stop)
        
        legs_to_check = []
        for temp_leg in leg_list: 
            # since we want to check forward and backward
            for cur_leg in [temp_leg,
                            temp_leg._replace(start=temp_leg.stop, 
                                           stop=temp_leg.start)]:
                if cur_leg.start==cur_start:
                    if cur_leg.stop==stop:
                        if _callback is not None:
                            _callback(cur_start, stop)
                        return cur_distance+cur_leg.distance
                    elif cur_leg.stop not in visited_legs:
                        legs_to_check += [cur_leg]

        for c_leg in legs_to_check:
            # we want to avoid visiting any of the stops on this iteration again
            visited_legs.add(c_leg.stop)

        for c_leg in legs_to_check:
            # this goes all the way down the before returning
            new_dist = cur_distance+c_leg.distance
            if (maximum_distance is None) or (maximum_distance is not None and (new_dist<maximum_distance)):
                start_stack += [[new_dist, c_leg.stop, c_leg]]
    return None              
# a utility function to compare the methods
from time import time
def timeit(f, number = 5): # homemade sloppy timeit
    start = time()
    for i in range(number):
        f()
    return (time()-start)/number

def compare_dfs_and_bfs(start, stop, max_dist = 100000):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 10))
    ax1.axis('off')
    ax2.axis('off')
    def _dfs_callback(x1, x2):
        show_stop(x1, ax=ax1)
        show_stop(x2, ax=ax1)
        show_leg(x1, x2, ax=ax1)
    tree_calc_total_distance(start, stop, 
                             unique_legs, 
                             mode='dfs', 
                             maximum_distance=max_dist,
                            _callback = _dfs_callback)
    time_search = lambda mode: timeit(lambda : tree_calc_total_distance(start, stop, unique_legs, mode=mode, maximum_distance=max_dist), number=1)
    
    ax1.set_title('Depth First - Compute: %2.2fs' % time_search('dfs'))
    def _bfs_callback(x1, x2):
        show_stop(x1, ax=ax2)
        show_stop(x2, ax=ax2)
        show_leg(x1, x2, ax=ax2)
    tree_calc_total_distance(start, stop, 
                             unique_legs, 
                             mode='bfs', 
                             maximum_distance=max_dist,
                            _callback = _bfs_callback)
    ax2.set_title('Breadth First - Compute:%2.2fs' % time_search('bfs'))
compare_dfs_and_bfs('Buchs AG, Wynenfeld', 
                    'Buchs AG, Heuweg', max_dist = 2000)
compare_dfs_and_bfs('Zürich, Beckenhof', 
                    'Zürich, Milchbuck', max_dist = 1500)
compare_dfs_and_bfs('Zürich, Beckenhof', 
                    'Zürich, Oerlikon', max_dist = 3000)
compare_dfs_and_bfs('Zürich HB', 
                    'Luzern', max_dist = 100000)
compare_dfs_and_bfs('Uster, Strick', 'Gossau ZH, Mitteldorf', max_dist = 1500)
import sys
sys.setrecursionlimit(200) # code is sloppy so lets not go nuts
def r_calc_total_distance(start, stop, leg_list, 
                        visited_legs = None, 
                        _callback = None):
    if visited_legs is None:
        visited_legs = set([start])
    
    legs_to_check = []
    for temp_leg in leg_list: 
        # since we want to check forward and backward
        for cur_leg in [temp_leg,
                        temp_leg._replace(start=temp_leg.stop, 
                                       stop=temp_leg.start)]:
            if cur_leg.start==start:
                if cur_leg.stop==stop:
                    if _callback is not None:
                        _callback(start, stop)
                    return cur_leg.distance
                elif cur_leg.stop not in visited_legs:
                    legs_to_check += [cur_leg]
    
    if len(legs_to_check)<1:
        return None
    
    for c_leg in legs_to_check:
        # we want to avoid visiting any of the stops on this iteration again
        visited_legs.add(c_leg.stop)
    
    for c_leg in legs_to_check:
        if _callback is not None:
            _callback(start, c_leg.start)
            _callback(c_leg.start, c_leg.stop)
        # this goes all the way down the before returning
        next_dist = r_calc_total_distance(c_leg.stop, 
                                        stop, 
                                        leg_list=leg_list, 
                                        visited_legs=visited_legs,
                                       _callback = _callback)
        if next_dist is not None:
            return c_leg.distance+next_dist
    return None              
fig, ax1 = plt.subplots(1, 1, figsize = (20, 10))
def my_callback(x1, x2):
    show_stop(x1, ax=ax1)
    show_stop(x2, ax=ax1)
    show_leg(x1, x2, ax=ax1)
try:
    out_dist = r_calc_total_distance('Zürich, Beckenhof', 
                               'Luzern', 
                               unique_legs, 
                              _callback = my_callback)
except RecursionError as e:
    print('Not connected, too deep', e)
