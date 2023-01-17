# Import all the libraries which will be used in the exploration process

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for data visualisations

import matplotlib.pyplot as plt # for data plotting and visualisations

from matplotlib.pyplot import figure, pie

from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.patches import PathPatch

import math



import bq_helper

from bq_helper import BigQueryHelper



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pylab import rcParams

rcParams['figure.figsize'] = 40, 10
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="chicago_crime")

bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
select_query = """SELECT date,district,primary_type,location_description,ward,arrest,domestic,community_area,year,latitude,longitude,location

            FROM `bigquery-public-data.chicago_crime.crime`

            LIMIT 300000"""

crime_data = chicago_crime.query_to_pandas_safe(select_query)
month_year_frame = pd.DataFrame(columns=[])



for i in range(0,crime_data.shape[0]):

    month = crime_data.iloc[i].date.strftime("%b")

    year = str(crime_data.iloc[i].year)

        

    try:

        get_count = month_year_frame.at[month, year]

        if np.isnan(get_count):

            month_year_frame.at[month, year] = 1

        else:

            month_year_frame.at[month, year] = get_count+1

    except (ValueError,KeyError):

        month_year_frame.at[month, year] = 1



month_year_frame.index = pd.CategoricalIndex(month_year_frame.index, 

                               categories=['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec'])

month_year_frame = month_year_frame.sort_index()

month_year_frame = month_year_frame.reindex(sorted(month_year_frame.columns), axis=1)



sns.heatmap(month_year_frame, cmap='gist_ncar')
corresponding_arrest = pd.DataFrame(columns=[])



for i in range(0,crime_data.shape[0]):

    month = crime_data.iloc[i].date.strftime("%b")

    year = str(crime_data.iloc[i].year)



    if crime_data.iloc[i].arrest:        

        try:

            get_count = corresponding_arrest.at[month, year]



            if np.isnan(get_count):

                corresponding_arrest.at[month, year] = 1

            else:

                corresponding_arrest.at[month, year] = get_count+1

        except (ValueError,KeyError):

            corresponding_arrest.at[month, year] = 1

            

corresponding_arrest.index = pd.CategoricalIndex(corresponding_arrest.index, 

                               categories=['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec'])

corresponding_arrest = corresponding_arrest.sort_index()

corresponding_arrest = corresponding_arrest.reindex(sorted(corresponding_arrest.columns), axis=1)
fig = plt.figure()

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



sns.heatmap(month_year_frame, cmap="gist_ncar",ax=ax1)

sns.heatmap(corresponding_arrest, cmap="gist_ncar",ax=ax2)
district_list = []



for i in range(0,crime_data.shape[0]):

    district = crime_data.iloc[i].district

    arrest = crime_data.iloc[i].arrest

    get_index = -1

    

    for j in range(0, len(district_list)):

        if (district_list[j][0] == district):

            get_index = j

            if arrest:

                district_list[j][1]+=1

            else:

                district_list[j][2]+=1

    

    if get_index == -1:

        if arrest:

            district_list.append([district, 1, 0])

        else:

            district_list.append([district, 0, 1])





get_district = pd.DataFrame(columns=['district','arrest','not_arrest'], data=district_list) 

get_district['Total'] = get_district.apply(lambda x: x.arrest+x.not_arrest, axis=1)



sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(10, 15))



# Load the example car crash dataset



# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(x="Total", y="district", data=get_district,

            label="Total", color="b", orient='h')



sns.set_color_codes("muted")

sns.barplot(x="arrest", y="district", data=get_district,

            label="Arrest", color="b", orient='h')



ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(ylabel="",

       xlabel="Total cases vs arrest")

sns.despine(left=True, bottom=True)
get_type = []



for i in range(0,crime_data.shape[0]):

    primary = crime_data.iloc[i].primary_type

    get_index = -1

    

    for j in range(0, len(get_type)):

        if (get_type[j][0] == primary):

            get_index = j

            get_type[j][1]+=1

    

    if get_index == -1:

        get_type.append([primary, 1])



type_data = pd.DataFrame(columns=['Type', 'count'], data=get_type)

fig1, ax1 = plt.subplots()

fig1.set_size_inches(18.5, 10.5)

ax1.pie(type_data['count'], labels=type_data['Type'], autopct='%1.1f%%',startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
crime_data['month'] = crime_data.apply(lambda x: x.date.strftime("%b"), axis=1)
year_range = {

    1: [2001,2005],

    2: [2006,2010],

    3: [2011,2015],

    4: [2016,2020]

}



def get_year_ref(year):

    for index, dic in enumerate(year_range):

        gap = year_range[dic]

        if year >= gap[0] and year <= gap[1]:

            return dic

        

get_month_year = []



for i in range(0,crime_data.shape[0]):

    row = crime_data.iloc[i]

    get_index = -1

    get_year_index = get_year_ref(row.year)

    

    for j in range(0, len(get_month_year)):

        if get_month_year[j][0] == row.month:

            get_index = j

            get_month_year[j][get_year_index]+=1

    

    if get_index == -1:

        create_arr = [0] * 5

        create_arr[0] = row.month

        create_arr[get_year_index] = 1

        get_month_year.append(create_arr)



month_wise_crime = pd.DataFrame(columns=['month','2001-2005','2006-2010','2011-2015','2016-2020'], data=get_month_year) 

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 

          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

month_wise_crime['month'] = pd.Categorical(month_wise_crime['month'], categories=months, ordered=True)

month_wise_crime.sort_values('month', inplace=True)
rcParams['figure.figsize'] = 20, 10

sns.barplot(data=month_wise_crime)
fig = plt.figure()

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)



sns.barplot(x='month',y='2001-2005', data=month_wise_crime, ax=ax1)

sns.barplot(x='month',y='2006-2010', data=month_wise_crime, ax=ax2)

sns.barplot(x='month',y='2011-2015', data=month_wise_crime, ax=ax3)

sns.barplot(x='month',y='2016-2020', data=month_wise_crime, ax=ax4)
fig = plt.figure(figsize=(22, 12))

ax = fig.add_subplot(111)

cm = plt.get_cmap('Reds')



district_val = pd.DataFrame(crime_data.district.value_counts().reset_index().values, columns=["district", "count"])



m = Basemap(projection='lcc', resolution='l', 

            lat_0=41.867779, lon_0=-87.638403,

            width=0.06E6, height=0.06E6)

m.drawmapboundary()



m.readshapefile('../input/geo_export_4ea0a4fd-5ba9-4f3f-bb35-ccd16bfc2ff9', 

                    name='world', 

                    drawbounds=True, 

                    color='gray')



for info,shape in zip(m.world_info, m.world):

    color = '#dddddd'

    for i in range(0,len(district_val)):

        if str(math.ceil(district_val.iloc[i].district)) == info['dist_num']:

            color =  cm(district_val.iloc[i]['count'] / district_val['count'].sum())

            break





    patches = [Polygon(np.array(shape), True)]

    pc = PatchCollection(patches)

    pc.set_facecolor(color)

    ax.add_collection(pc)