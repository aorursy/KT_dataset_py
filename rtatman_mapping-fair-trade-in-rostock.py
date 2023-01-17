import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import contextily as ctx
import matplotlib.pyplot as pyplot
import seaborn as sns

fair_trade = pd.read_csv("../input/fair-trade.csv")
# get map of location "rostock"
# TODO: figure out how to zoom in on just the city
loc = ctx.Place('rostock')
# plot fair trade locations in Rostock
(ctx.plot_map(loc).
     scatter(fair_trade['longitude'],
             fair_trade['latitude'])
)
# plot fair trade locations in Rostock, with 
# different kinds of venues different color

# method from this stack overflow question:
# https://stackoverflow.com/questions/26139423/plot-different-color-for-different-categorical-levels-using-matplotlib

# Unique category labels: 'D', 'F', 'G', ...
color_labels = fair_trade['art'].unique()

# List of RGB triplets
rgb_values = sns.color_palette("Set2", 8)

# Map label to RGB
color_map = dict(zip(color_labels, rgb_values))

# Finally use the mapped values
ctx.plot_map(loc).scatter(fair_trade['longitude'],
                          fair_trade['latitude'],
                          c=fair_trade['art'].map(color_map))