import pandas as pd



import datetime



import numpy as np

import osmnx as ox

import networkx as nx



from sklearn.neighbors import KDTree

import folium



import matplotlib.pyplot as plt

%matplotlib inline
# Read data from csv

speeds_df = pd.read_csv('../input/speeds_01_june_2018.csv')



# Delete Unamed column

del speeds_df['Unnamed: 0']



speeds_df.head()
# Make a copy of the speeds_df

avg_speed_edge_df = speeds_df.copy()



# Get the average speed on the link at a given time

avg_speed_edge_df = avg_speed_edge_df.groupby('edge_id', as_index=False).mean()



# Insert a column for the direction

avg_speed_edge_df.insert(2, "direction", "", allow_duplicates = False)



# Get the direction from the speed_df

for edge in avg_speed_edge_df.edge_id:

    avg_speed_edge_df.loc[avg_speed_edge_df['edge_id'] == edge, "direction"] = speeds_df.loc[speeds_df['edge_id'] == edge, "direction"].iloc[0]

    

avg_speed_edge_df.head()

bin_labels = np.arange(2.5, 247.5, 5)

cut_bins = np.arange(0, 250, 5)



# Assign bin for each speed into 5km/h spaced bins

for col in avg_speed_edge_df.columns[3:]:

    avg_speed_edge_df[col] = pd.cut(avg_speed_edge_df[col], bins=cut_bins, labels=bin_labels)



avg_speed_edge_df.head()
fig = plt.figure(num=None, figsize=(8, 3), dpi=300, facecolor='w', edgecolor='k')

ax = plt.axes()

ax.tick_params(labelsize=12)

x = avg_speed_edge_df.iloc[0][3:].index

y = avg_speed_edge_df.iloc[0][3:]

ax.plot(x, y, '-o');

plt.margins(x=0, y=0.4)

plt.ylabel('Average speed of network (km/h)', size = 12)

plt.xlabel('Hour of day', size = 12)

morning_df = avg_speed_edge_df.loc[:, "edge_id":"direction"]



# Get the speeds between 6:30am - 10:30am

for col in avg_speed_edge_df.columns[42:67]:

    morning_df[pd.to_datetime(col).strftime("%-H:%M")] = pd.Series(avg_speed_edge_df[col])



morning_df.head()
evening_df = avg_speed_edge_df.loc[:, "edge_id":"direction"]



# Get the speeds between 3pm - 7pm

for col in avg_speed_edge_df.columns[93:118]:

    evening_df[pd.to_datetime(col).strftime("%-H:%M")] = pd.Series(avg_speed_edge_df[col])



evening_df.head()
fig = plt.figure(num=None, figsize=(10, 3), dpi=300, facecolor='w', edgecolor='k')

ax = plt.axes()

ax.tick_params(labelsize=12)

x = morning_df.columns[3:]

plt.xticks(rotation=90)

y = morning_df.iloc[0][3:]

ax.plot(x, y, '-o');

plt.margins(x=0, y=0.4)

plt.ylabel(f'Average speed of edge {morning_df.iloc[0].edge_id} (km/h)', size = 12)

plt.xlabel('Hour of day', size = 12)
for idx, edge in enumerate(morning_df.edge_id):

    fig = plt.figure(num=None, figsize=(10, 3), dpi=300, facecolor='w', edgecolor='k')

    ax = plt.axes()

    direction = morning_df.iloc[idx]['direction']

    if direction == "D02":

        direction = "Citybound"

    elif direction == "D01":

        direction = "Southbound"

    else:

        direction = "Roundabout"

    plt.title(f'Edge {edge} speeds during morning peak (06:30 - 10:30) ({direction})')

    ax.tick_params(labelsize=12)

    x = morning_df.columns[3:]

    plt.xticks(rotation=90)

    y = morning_df.iloc[idx][3:]

    ax.plot(x, y, '-o');

    plt.margins(x=0, y=0.4)

    plt.ylabel('Average speed (km/h)', size = 12)

    plt.xlabel('Time', size = 12)

    plt.show()

    plt.close()
for idx, edge in enumerate(evening_df.edge_id):

    fig = plt.figure(num=None, figsize=(10, 3), dpi=300, facecolor='w', edgecolor='k')

    ax = plt.axes()

    direction = evening_df.iloc[idx]['direction']

    if direction == "D02":

        direction = "Citybound"

    elif direction == "D01":

        direction = "Southbound"

    else:

        direction = "Roundabout"

    plt.title(f'Edge {edge} speeds during evening peak (15:00 - 19:00) ({direction})')

    ax.tick_params(labelsize=12)

    x = evening_df.columns[3:]

    plt.xticks(rotation=90)

    y = evening_df.iloc[idx][3:]

    ax.plot(x, y, '-o');

    plt.margins(x=0, y=0.4)

    plt.ylabel('Average speed (km/h)', size = 12)

    plt.xlabel('Time', size = 12)

    plt.show()

    plt.close()