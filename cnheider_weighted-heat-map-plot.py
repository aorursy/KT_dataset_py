import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gmaps

gmaps.configure(api_key="####insert-here####")
df = pd.read_csv("../input/ghcn-m-v1.csv", na_values=[-9999]).fillna(0)
def get_sign(direction):

    return {

        'N': 1.,

        'E': 1.,

        'S': -1.,

        'W': -1

    }[direction]
def extract_lons(list_of_str_coords):

    lol = [x.split('_')[1:] for x in list_of_str_coords]

    return [(float(y[0])+float(y[1][:-1]))/2.*get_sign(y[1][-1]) for y in lol]



def extract_lats(list_of_str_coords):

    lol = [x.split('-') for x in list_of_str_coords]

    return [(float(y[0])+float(y[1][:-1]))/2.*get_sign(y[1][-1]) for y in lol]
lats = extract_lats(df.lat)

lons = extract_lons(df.columns[3:])

df.lat = lats

df.columns = ['year','month','lat'] + lons
def heatmap_layer(values, val_max, color):

    heatmap_layer = gmaps.WeightedHeatmap(data = values)

    heatmap_layer.max_intensity = val_max

    heatmap_layer.point_radius = 10

    heatmap_layer.dissipating = False

    heatmap_layer.gradient = [(0, 0, 0, 0.0),color]

    return heatmap_layer
def gmap_plot(year_month):

    pivot = year_month.pivot_table(index='lat')

    del pivot['month']

    del pivot['year']

    melt = pd.melt(pivot.reset_index(), id_vars=['lat'], var_name='lon', value_name='val')

    pos_val = melt[melt.val >= 0.].values

    pos_val_list = [x for x in pos_val[:,2].tolist()]

    pos_max = max(pos_val_list)

    neg_val = melt[melt.val <= 0.].values

    neg_val_list = [abs(x) for x in neg_val[:,2].tolist()]

    neg_max = max(neg_val_list)

    pos_tuples = [(x[0][0],x[0][1],x[1]) for x in zip(pos_val, pos_val_list)]

    neg_tuples = [(x[0][0],x[0][1],x[1]) for x in zip(neg_val, neg_val_list)]

    heatmap_layer_neg = heatmap_layer(neg_tuples, neg_max, (0, 0, 255))

    heatmap_layer_pos = heatmap_layer(pos_tuples, pos_max, (255, 0, 0))

    gmap = gmaps.Map()

    gmap.add_layer(heatmap_layer_neg)

    gmap.add_layer(heatmap_layer_pos)

    return gmap
years = df.groupby('year')
jan1880 = years.get_group(1880).groupby('month').get_group(1)

gmap_plot(jan1880)
jan2016 = years.get_group(2016).groupby('month').get_group(1)

gmap_plot(jan2016)