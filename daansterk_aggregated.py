# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



import plotly.plotly as py

import plotly.graph_objs as go



import seaborn as sns

sns.set_style('whitegrid')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
t_data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1',

                    usecols=[0, 1, 2, 3, 7, 9, 11, 12, 13, 14, 15, 19, 20, 21, 26, 28, 29, 34, 35, 36, 58, 68, 69, 81, 98, 101, 104, 134]) 



# Fill empty cells with proper value

t_data['nkill'] = t_data['nkill'].fillna(0).astype(int)

t_data['nwound'] = t_data['nwound'].fillna(0).astype(int)
ax = sns.countplot(y="attacktype1_txt", data=t_data)

ax.set_xlabel("Amount of type")

ax.set_ylabel("Targettype")
# Rounds the long- and latitude to a number withouth decimals, groups them on long- and latitude and counts the amount of attacks.

df_coords = t_data.round({'longitude':0, 'latitude':0}).groupby(["longitude", "latitude"]).size().to_frame(name = 'count').reset_index()

sns.jointplot(x='longitude', y='latitude', data=df_coords, kind="hex", color="#4CB391", size=15, stat_func=None, edgecolor="#EAEAF2", linewidth=.2)

plt.title('Amount of terrorist attacks per rounded coordinates')
fig, ax = plt.subplots(figsize=(15,15))

sns.countplot(x="iday", data=df_day_coords, ax=ax, palette=sns.cubehelix_palette(15, start=.3, rot=.3))

ax.set_xlabel('Day of the month')

ax.set_ylabel('Amount of terrorist attacks')
fig, axs = plt.subplots(nrows=12)

fig.set_size_inches(15, 100, forward=True)



for i in range(1,13):

    monthly_data = df_day_coords[df_day_coords['imonth'] == i]

    sns.countplot(x="iday", data=monthly_data, hue="success", ax=axs[i-1])

    axs[i-1].set_xlabel('Day of the month')

    axs[i-1].set_ylabel('Amount of terrorist attacks')
region_dictionary = {1: 'North America', 2: 'Central America & Carribean', 3: 'South America',

                     4: 'East Asia', 5: 'Southeast Asia', 6: 'South Asia', 7: 'Central Asia',

                     8: 'Western Europe', 9: 'Eastern Europe', 10: 'Middle East and North Africa',

                     11: 'Sub-Saharan Africa', 12: 'Australasia and Oceania'}



def generate_graph(region_grouped_success, region_grouped_failure,region_data,region_number):

    #setting formatting

    fig = plt.figure(figsize = (15,6))

    ax1 = fig.add_subplot(1,2,1)

    

    ax1.set(title = '#Attacks region %s ' % region_dictionary[region_number],

            ylabel = 'Attack count', xlabel = 'year')



    #entering data

    ax1.plot(region_grouped_success.index, region_grouped_success.eventid, label = 'Successfull attacks' )

    ax1.plot(region_grouped_failure.index, region_grouped_failure.eventid, label = 'Failed attacks' )

    #add legend

    ax1.legend(loc = 'upper left', frameon = True, edgecolor = 'black',bbox_to_anchor =(1.1,0.1))

#    ax1.plot(region_data.index, region_data.eventid ) # sum successfull and failure data



def group_data(region_data, region_number):

    region_data_success = region_data[(region_data.success == 1)] #filter on success

    region_data_failure = region_data[(region_data.success == 0)] #filter on failure

    

    generate_graph(region_data_success.groupby('iyear').count(),

                   region_data_failure.groupby('iyear').count(),

                   region_data,

                   region_number) #create line plot for region that is grouped by year



        

def select_region(region_list):

        for region_number in region_dictionary:

            group_data(t_data[(t_data.region == region_number)],

                          region_number) #for each region group data by year



        

select_region(region_dictionary)
def multi_graph(result,result_list, xmin, xmax, ymin, ymax):

    fig2, ax2 = plt.subplots(figsize = (15,8))

    number = 1 #the for-loop in append_list processes the regions in order from 1 to 12

    for j in result_list:

        #print(result_list)

        #if j.iyear >= xmin and j.iyear <= xmax: #hoe axis grenzen hiermee filteren?

        ax2.plot(j.index, j.eventid, label = '%s ' % region_dictionary[number] )

        number += 1



    plt.xlim([xmin,xmax])

    plt.ylim([ymin,ymax])

    plt.xlabel('year')

    plt.ylabel('number of attacks')

    plt.title(result)

    ax2.legend(loc = 'center', frameon = True, edgecolor = 'black',bbox_to_anchor =(1.2,0.4))





success_list = []

failure_list = []



for i in region_dictionary:

    region_data = t_data[(t_data.region == i)]

    region_data_success = region_data[(region_data.success == 1)]

    region_data_failure = region_data[(region_data.success == 0)]

    region_grouped_success = region_data_success.groupby('iyear').count()

    region_grouped_failure = region_data_failure.groupby('iyear').count()



    success_list.append(region_grouped_success)

    failure_list.append(region_grouped_failure)



multi_graph('Successes',success_list, 1970, 2011, 0, 2100)

multi_graph('Successes',success_list, 2012, 2016, 0, 6500)

multi_graph('Failures',failure_list, 1970, 2011, 0, 200)

multi_graph('Failures',failure_list, 2012, 2016, 0, 1300)