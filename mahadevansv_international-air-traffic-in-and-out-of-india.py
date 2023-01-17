# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df_traffic_monthly = pd.read_csv("../input/international-air-traffic-from-and-to-india/Airlinewise Monthly International Air Traffic To And From The Indian Territory.csv")



df_traffic_passengers = df_traffic_monthly[(df_traffic_monthly['PASSENGERS FROM INDIA'] > 0) | (df_traffic_monthly['PASSENGERS TO INDIA'] > 0)]



df_traffic_passengers_yearly = df_traffic_passengers.groupby(['AIRLINE NAME', 'YEAR'])['PASSENGERS TO INDIA', 'PASSENGERS FROM INDIA'].sum().reset_index()



df_traffic_passengers_yearly['TOTAL TRAFFIC'] = df_traffic_passengers_yearly['PASSENGERS TO INDIA'] + df_traffic_passengers_yearly['PASSENGERS FROM INDIA']



#df_traffic_passengers_from = df_traffic_passengers.groupby(['AIRLINE NAME', 'YEAR'])[].sum().nlargest(20).reset_index(name='FROM COUNT')



#df_traffic_passengers_combined = pd.merge(df_traffic_passengers_to, df_traffic_passengers_from, on='AIRLINE NAME')

#df_traffic_passengers_combined['TOTAL TRAFFIC'] = df_traffic_passengers_combined['FROM COUNT'] + df_traffic_passengers_combined['TO COUNT']

#df_traffic_passengers_combined['OUTBOUND TO INBOUND RATIO'] = df_traffic_passengers_combined['FROM COUNT'] / df_traffic_passengers_combined['TO COUNT']

#df_traffic_passengers_combined.plot(x='AIRLINE NAME', y='TOTAL TRAFFIC', kind='pie', legend=False, autopct='%1.1f%%', labels=df_traffic_passengers_to['AIRLINE NAME'], figsize=(10, 10), title='Top 10 Carrying passengers to India')



df_traffic_2015 = df_traffic_passengers_yearly[df_traffic_passengers_yearly['YEAR'] == 2015]

df_traffic_2015_top10 = df_traffic_2015[df_traffic_2015['TOTAL TRAFFIC'].isin(df_traffic_2015['TOTAL TRAFFIC'].nlargest(10))]



df_traffic_2016 = df_traffic_passengers_yearly[df_traffic_passengers_yearly['YEAR'] == 2016]

df_traffic_2016_top10 = df_traffic_2016[df_traffic_2016['TOTAL TRAFFIC'].isin(df_traffic_2016['TOTAL TRAFFIC'].nlargest(10))]



df_traffic_2017 = df_traffic_passengers_yearly[df_traffic_passengers_yearly['YEAR'] == 2017]

df_traffic_2017_top10 = df_traffic_2017[df_traffic_2017['TOTAL TRAFFIC'].isin(df_traffic_2017['TOTAL TRAFFIC'].nlargest(10))]



fig = plt.figure()

fig.subplots_adjust(hspace=1.2, wspace=.5)



ax = plt.subplot(221)



df_traffic_2015_top10.plot(ax=ax, x='AIRLINE NAME', y='TOTAL TRAFFIC', kind='bar', figsize=(12, 8), title='2015 Traffic')

ax = plt.subplot(222)

df_traffic_2016_top10.plot(ax=ax, x='AIRLINE NAME', y='TOTAL TRAFFIC', kind='bar', title='2016 Traffic')

ax = plt.subplot(223)

df_traffic_2017_top10.plot(ax=ax, x='AIRLINE NAME', y='TOTAL TRAFFIC', kind='bar', title='2017 Q1 Traffic')
fig, axes = plt.subplots(nrows=1, ncols=2)



df_qtrly = df_traffic_passengers.groupby(['QUARTER'])['PASSENGERS TO INDIA', 'PASSENGERS FROM INDIA'].sum().reset_index()

df_qtrly.plot(x='QUARTER', y=['PASSENGERS TO INDIA', 'PASSENGERS FROM INDIA'], kind='bar', ax=axes[0], figsize=(12, 7))



df_monthly = df_traffic_passengers.groupby(['MONTH'])['PASSENGERS TO INDIA', 'PASSENGERS FROM INDIA'].sum().reset_index()

df_monthly.plot(x='MONTH', y=['PASSENGERS TO INDIA', 'PASSENGERS FROM INDIA'], kind='bar', ax=axes[1], figsize=(12, 7))

df_yearly = df_traffic_passengers.groupby(['YEAR', 'QUARTER'])['PASSENGERS TO INDIA', 'PASSENGERS FROM INDIA'].sum().reset_index()

df_yearly = df_yearly[df_yearly['QUARTER'] == 'Q1']

df_yearly.plot(x=['YEAR', 'QUARTER'], y=['PASSENGERS TO INDIA', 'PASSENGERS FROM INDIA'], kind='line', figsize=(10, 10))

df_yearly
df_qtrly_airlines = df_traffic_passengers.groupby(['QUARTER', 'AIRLINE NAME'])['PASSENGERS TO INDIA', 'PASSENGERS FROM INDIA'].sum().reset_index()

df_qtrly_airlines['TOTAL'] = df_qtrly_airlines['PASSENGERS TO INDIA'] + df_qtrly_airlines['PASSENGERS FROM INDIA'] 



df_airlines_q1 = df_qtrly_airlines[df_qtrly_airlines['QUARTER'] == 'Q1']

df_q1_airlines_top10 = df_airlines_q1[df_airlines_q1['TOTAL'].isin(df_airlines_q1['TOTAL'].nlargest(10))]



df_airlines_q2 = df_qtrly_airlines[df_qtrly_airlines['QUARTER'] == 'Q2']

df_q2_airlines_top10 = df_airlines_q2[df_airlines_q2['TOTAL'].isin(df_airlines_q2['TOTAL'].nlargest(10))]



df_airlines_q3 = df_qtrly_airlines[df_qtrly_airlines['QUARTER'] == 'Q3']

df_q3_airlines_top10 = df_airlines_q3[df_airlines_q3['TOTAL'].isin(df_airlines_q3['TOTAL'].nlargest(10))]



df_airlines_q4 = df_qtrly_airlines[df_qtrly_airlines['QUARTER'] == 'Q4']

df_q4_airlines_top10 = df_airlines_q4[df_airlines_q4['TOTAL'].isin(df_airlines_q4['TOTAL'].nlargest(10))]



fig, axes = plt.subplots(nrows=2, ncols=2)



df_q1_airlines_top10.plot(x='AIRLINE NAME', y=['TOTAL'], kind='bar', ax=axes[0,0], figsize=(15, 20), title = 'Q1 Traffic')

df_q2_airlines_top10.plot(x='AIRLINE NAME', y=['TOTAL'], kind='bar', ax=axes[0,1], title = 'Q2 Traffic')

df_q3_airlines_top10.plot(x='AIRLINE NAME', y=['TOTAL'], kind='bar', ax=axes[1,0], title = 'Q3 Traffic')

df_q4_airlines_top10.plot(x='AIRLINE NAME', y=['TOTAL'], kind='bar', ax=axes[1,1], title = 'Q4 Traffic')
df_citywise = pd.read_csv("../input/international-air-traffic-from-and-to-india/Citypairwise Quarterly International  Air Traffic To And From The Indian Territory.csv", error_bad_lines=False, warn_bad_lines=False)

df_citywise = df_citywise.iloc[:, :-3]
df_citywise['TOTAL TRAFFIC'] = df_citywise['PASSENGERS FROM CITY1 TO CITY2'] + df_citywise['PASSENGERS FROM CITY2 TO CITY1']

df_citywise.head(10)
df_citywise_top100 = df_citywise[df_citywise['TOTAL TRAFFIC'].isin(df_citywise['TOTAL TRAFFIC'].nlargest(100))]

df_citywise_top100.head(20)
df_citywise_cumulative_traffic = df_citywise.groupby(['CITY1', 'CITY2'])['TOTAL TRAFFIC'].sum().nlargest(15).reset_index()

highest_traffic = np.max(df_citywise['TOTAL TRAFFIC'])

uniqueCity1, city1Ints = np.unique(df_citywise_cumulative_traffic['CITY1'], return_inverse=True)

uniqueCity2, city2Ints = np.unique(df_citywise_cumulative_traffic['CITY2'], return_inverse=True)





df_citywise_cumulative_traffic['TOTAL TRAFFIC SCALED'] = (df_citywise_cumulative_traffic['TOTAL TRAFFIC'] / highest_traffic) * 100





fig = plt.figure(figsize=(20, 20))

ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.scatter(city1Ints, city2Ints, df_citywise_cumulative_traffic['TOTAL TRAFFIC'], s=df_citywise_cumulative_traffic['TOTAL TRAFFIC SCALED'], c='b')

ax.set(xticks=range(len(uniqueCity1)), xticklabels=uniqueCity1,

       yticks=range(len(uniqueCity2)), yticklabels=uniqueCity2) 

plt.show()



df_citywise_cumulative_traffic.head(10)

from mpl_toolkits.basemap import Basemap

from matplotlib.colors import Normalize, LinearSegmentedColormap, PowerNorm



def plot_map(in_filename, color_mode='screen',

             out_filename='flights_map_mpl.png', absolute=False):

    if color_mode == 'screen':

        bg_color = (0.0, 0.0, 0, 1.0)

        coast_color = (204/255.0, 0, 153/255.0, 0.7)

        color_list = [(0.0, 0.0, 0.0, 0.0),

                      (204/255.0, 0, 153/255.0, 0.6),

                      (255/255.0, 204/255.0, 230/255.0, 1.0)]

    else:

        bg_color = (1.0, 1.0, 1.0, 1.0)

        coast_color = (10.0/255.0, 10.0/255.0, 10/255.0, 0.8)

        color_list = [(1.0, 1.0, 1.0, 0.0),

                      (255/255.0, 204/255.0, 230/255.0, 1.0),

                      (204/255.0, 0, 153/255.0, 0.6)

                      ]



    # define the expected CSV columns

    CSV_COLS = ('dep_lat', 'dep_lon', 'arr_lat', 'arr_lon',

                'nb_flights', 'CO2')



    routes = pd.read_csv(in_filename, names=CSV_COLS, na_values=['\\N'],

                         sep=',', skiprows=1)





    num_routes = len(routes.index)



    # normalize the dataset for color scale

    norm = PowerNorm(0.3, routes['nb_flights'].min(),

                     routes['nb_flights'].max())

    # norm = Normalize(routes['nb_flights'].min(), routes['nb_flights'].max())



    # create a linear color scale with enough colors

    if absolute:

        n = routes['nb_flights'].max()

    else:

        n = num_routes

    cmap = LinearSegmentedColormap.from_list('cmap_flights', color_list,

                                             N=n)

    # create the map and draw country boundaries

    plt.figure(figsize=(27, 20))

    m = Basemap(projection='mill', lon_0=0)

    

    m.drawcoastlines(color=coast_color, linewidth=1.25)

    m.fillcontinents(color=bg_color, lake_color=bg_color)

    m.drawmapboundary(fill_color=bg_color)



    # plot each route with its color depending on the number of flights

    for i, route in enumerate(routes.sort_values(by='nb_flights',

                              ascending=True).iterrows()):

        route = route[1]

        if absolute:

            color = cmap(norm(int(route['nb_flights'])))

        else:

            color = cmap(i * 1.0 / num_routes)



        line, = m.drawgreatcircle(route['dep_lon'], route['dep_lat'],

                                  route['arr_lon'], route['arr_lat'],

                                  linewidth=0.5, color=color)

        # if the path wraps the image, basemap plots a nasty line connecting

        # the points at the opposite border of the map.

        # we thus detect path that are bigger than 30km and split them

        # by adding a NaN

        path = line.get_path()

        cut_point, = np.where(np.abs(np.diff(path.vertices[:, 0])) > 30000e3)

        if len(cut_point) > 0:

            cut_point = cut_point[0]

            vertices = np.concatenate([path.vertices[:cut_point, :],

                                      [[np.nan, np.nan]],

                                      path.vertices[cut_point+1:, :]])

            path.codes = None  # treat vertices as a serie of line segments

            path.vertices = vertices



    # save the map

    plt.show()



plot_map('../input/airtrafficcoordinatesindia/flight_data.csv', 'screen', absolute=True)
