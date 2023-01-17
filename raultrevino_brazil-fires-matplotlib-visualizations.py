import pandas as pd

import matplotlib as mp

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.patches as mpatches
data = pd.read_csv("../input/forest-fires-in-brazil/amazon.csv", encoding='latin1')

data.head()
print("Numero de registros:"+str(data.shape[0]))

for column in data.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))
data['state'].unique()
data['month'].unique()
portuguese_months = data['month'].unique()

english_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',

       'August', 'September', 'October', 'November', 'December']



month_index = 0



for p_month in portuguese_months:

    data.loc[(data.month == p_month),'month']= english_months[month_index]

    month_index = month_index + 1

data['month'].unique()
data_total_fires = data.groupby('state')['number'].sum().to_frame().reset_index().sort_values(by=['number'],ascending=False)

data_total_fires.head()
color_map = mp.colors.LinearSegmentedColormap(

    "my_map",

    {"red": [(0, 1.0, 1.0),

                (1.0, .5, .5)],

        "green": [(0, 0.5, 0.5),

                  (1.0, 0, 0)],

        "blue": [(0, 0.50, 0.5),

                 (1.0, 0, 0)]

    }

)

data_normalizer = mp.colors.Normalize()

data_fires = data_total_fires['number']

# Colorize the graph based on likeability:

likeability_scores = np.array(data_fires.values.ravel())
%matplotlib inline

fig = plt.figure(figsize=(9, 6))

ax = fig.add_axes([0,0,1,1])

ax.set_title("Top 10 states with more fires since 1998 to 2017")

states = data_total_fires['state'][:10]

number_fires = data_total_fires['number'][:10]

barlist = ax.bar(states,number_fires,color=color_map(data_normalizer(likeability_scores)))

plt.show()
data_fires = data_total_fires['number']

data_fires = np.sort(data_fires)[::-1] #reverse array

 # Colorize the graph based on likeability:

likeability_scores = np.array(data_fires)



%matplotlib inline

fig = plt.figure(figsize=(10, 6))

ax = fig.add_axes([0,0,1,1])

ax.set_title("Next 11 states with more fires")

states = data_total_fires['state'][10:21]

number_fires = data_total_fires['number'][10:21]

ax.bar(states,number_fires,color=color_map(data_normalizer(likeability_scores)))

plt.show()
%matplotlib inline

fig = plt.figure(figsize=(10, 6))

ax = fig.add_axes([0,0,1,1])

ax.set_title("Top 5 regions with fires across time")



regions  = data_total_fires.state.unique()

patches = []

for region in regions[:5]:

    data_fires_in_region = data[(data['state'] == region)]

    data_fires_in_region = data_fires_in_region.groupby('year').sum().reset_index().sort_values(by=['year'])

    p = ax.plot(data_fires_in_region['year'].values.ravel(),data_fires_in_region['number'].values.ravel(),'o-')

    plot_color = p[0].get_color()

    patches.append(mpatches.Patch(color=plot_color,label=region))

    

plt.legend(handles=patches)

plt.xlabel("Year")

plt.ylabel("# Of fires")

plt.show()

    
data_total_fires_by_month = data.groupby('month')['number'].sum().to_frame().reset_index().sort_values(by=['number'],ascending=False)
data_normalizer = mp.colors.Normalize()

data_fires = data_total_fires_by_month['number']

# Colorize the graph based on likeability:

likeability_scores = np.array(data_fires)
%matplotlib inline

fig = plt.figure(figsize=(11, 6))

ax = fig.add_axes([0,0,1,1])

ax.set_title("Months with more fires")

months = data_total_fires_by_month['month']

number_fires = data_total_fires_by_month['number']

barlist = ax.bar(months,number_fires,color=color_map(data_normalizer(likeability_scores)))

plt.show()
data_total_fires = data.groupby('state')['number'].sum().to_frame().reset_index().sort_values(by=['number'],ascending=False)[:10]
%matplotlib inline

# Pie chart, where the slices will be ordered and plotted counter-clockwise:

fig1 = plt.figure(figsize=(11, 9))

ax1 = fig1.add_axes([0,0,1,1])

ax1.pie(data_total_fires['number'].values.ravel(), 

        labels=data_total_fires['state'].values.ravel(),

        autopct='%1.1f%%',

        shadow=True, startangle=90)



ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()