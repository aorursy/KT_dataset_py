# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import matplotlib.ticker as mtick

from matplotlib.dates import DateFormatter

import seaborn as sns; sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# importing data

data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

# other DB 

# data = pd.read_csv('../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200206.csv')





# round up data at the day level

data['Day'] = pd.to_datetime(data['Date']).dt.strftime('%Y/%m/%d')

data.drop(columns=['Last Update','Date'], inplace=True)



# If no data for the provice, replace by the whole country

rows = data[data['Province/State'].isnull()]['Province/State'].index 

data.loc[rows,'Province/State'] = data.loc[rows,'Country']



# get list of days in data set

days_list = list(data.sort_values('Day')['Day'].unique())

provinces_list = list(data['Province/State'].unique())





data.sort_values('Day').tail()
# data cleaning

# group by day and Provice; take the max of a given day (sometimes the day is updated several times on the same day)

new_data = data.groupby(['Day','Province/State']).agg('max')

# put 'Province/State' back in feature columns

new_data.reset_index(level=['Day','Province/State'], col_level=1, inplace=True)
# visualize grographical differences today



last_day = new_data['Day'].max()



# total cases today

total_confirmed = new_data[new_data['Day']==last_day]['Confirmed'].sum()



# get regional differences

region_data = new_data[new_data['Day']==last_day].groupby(['Province/State']).agg('sum')



region_data['Fraction of infections'] = region_data['Confirmed']/total_confirmed

region_data['Mortality rate'] = region_data['Deaths']/region_data['Confirmed']





region_data_top15 = region_data.sort_values('Confirmed', ascending=False).head(15)
import folium

import json

import branca.colormap as cm

latitude = 30.86166

longitude = 114.195397



china_provinces = '/kaggle/input/china-regions-map/china-provinces.json'



china_confirmed_colorscale = cm.linear.YlOrRd_09.scale(0, total_confirmed/20)

china_confirmed_series = region_data_top15['Confirmed']



def confirmed_style_function(feature):

    china_show = china_confirmed_series.get(str(feature['properties']['NAME_1']), None)

    return {

        'fillOpacity': 0.6,

        'weight': 0,

        'fillColor': 'white' if china_show is None else china_confirmed_colorscale(china_show)

    }



china_confirmed_map = folium.Map(location=[35.86, 104.19], zoom_start=4)



folium.TopoJson(

    json.load(open(china_provinces)),

    'objects.CHN_adm1',

    style_function=confirmed_style_function

).add_to(china_confirmed_map)



# localize Wuhan

folium.Marker(

    location=[30.5928, 114.3055],

    popup = folium.Popup('Wuhan.', show=True),

).add_to(china_confirmed_map)



#put color scale

china_confirmed_colorscale.caption = 'Confirmed cases'

china_confirmed_colorscale.add_to(china_confirmed_map)





china_confirmed_map

# check global cases (all regions together) 

global_data = new_data.groupby(['Day']).agg('sum')



# build the figure

fig = plt.figure("", figsize=(12, 12))



ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)



sns.barplot(x=global_data.index, y=global_data['Confirmed'], ci=None, ax=ax1)

ax1.set_title('Total confirmed cases')



#new_data['Confirmed'].plot.bar(stacked=True)

sns.barplot(x=global_data.index, y=global_data['Confirmed'].pct_change()*100, ci=None, ax=ax2)

# assign locator and formatter for the xaxis ticks.





# tilt the labels since they tend to be too long

fig.autofmt_xdate()

ax2.set_title('Relative increase [%]')

ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

ax2.set_ylabel('Increase relative to previous day [%]')

plt.show()
region_data_top15.style.set_properties(**{'text-align': 'left'})

region_data_top15.drop(['Sno'], axis=1).style.format({'Mortality rate': '{:,.2%}'.format, 'Fraction of infections': '{:,.2%}'.format})




fig = plt.figure("", figsize=(14, 14))



sns.set(font_scale=1.1)



ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)



palette1 = sns.color_palette("Paired", 8)

sns.barplot(y=region_data_top15.index, x='Confirmed', 

            data=region_data_top15, ax=ax1,

            palette = palette1)

ax1.set_title('First 15 region impacted by the coronavirus',size=16)





# exclude Hube2 = palette1

palette2 = palette1[1:] + palette1[:1] # move color representation by 1

sns.barplot(y=region_data_top15.drop(index=['Hubei']).index, x='Confirmed', 

            data=region_data_top15.drop(index=['Hubei']), ax=ax2,

           palette = palette2)

ax2.set_title('Zoom without Hubei province',size=16);



# all data with 'Others' label for the country turn out to be the cruise ship, replacing them

rows = data[data['Country']=='Others'].index

data.loc[rows,'Country'] = 'Diamond Princess cruise ship'



# remove China

data_other_countries = data[(data['Country'] != 'Mainland China') & (data['Country'] != 'China')]



# Get all country

data_other_countries = data_other_countries[data_other_countries['Day']==last_day].groupby(['Country']).agg('sum')

data_other_countries.drop(['Sno'], axis=1).sort_values('Confirmed', ascending=False).head(15)
# get the list of the first 15 provinces

top_15_provinces = region_data_top15.index.to_list()



# convert Day back to date format

new_data['Day'] = pd.to_datetime(new_data['Day'], infer_datetime_format=True) 





fig = plt.figure("", figsize=(14, 16))



sns.set(font_scale=1.1)



ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)



for province in top_15_provinces:

    

    # get days after which the number of cases got over 400

    if max(new_data[(new_data['Province/State'] == province)]['Confirmed']) > 400 :

        date_start = new_data[(new_data['Province/State'] == province) & (new_data['Confirmed'] > 400)]['Day'].iloc[0]

        realtive_days = (new_data[new_data['Province/State'] == province][['Day']]  - date_start).values.astype('timedelta64[D]')

        ax1.plot(realtive_days, new_data[new_data['Province/State'] == province][['Confirmed']])

        

        if province != 'Hubei' and province != 'Diamond Princess cruise ship':

            ax2.plot(realtive_days, new_data[new_data['Province/State'] == province][['Confirmed']].pct_change()*100, linestyle='--', marker = '*')

        else:

            ax2.plot(realtive_days, new_data[new_data['Province/State'] == province][['Confirmed']].pct_change()*100, linestyle='--', marker = '*', linewidth=3)

        

        

        

ax1.set_xlim([-5,15])    

ax1.set_ylim([0,3000])

ax1.legend(top_15_provinces)

ax1.set_xlabel('Days after more than 400 cases have been confirmed')

ax1.set_ylabel('confirmed cases')

ax1.set_title('Absolute numbers',size=16)



ax2.set_xlim([-5,15])    

ax2.set_ylim([0,50])

ax2.legend(top_15_provinces)

ax2.set_xlabel('Days after more than 400 cases have been confirmed')

ax2.set_ylabel('confirmed cases')

ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

ax2.set_ylabel('Increase relative to previous day [%]')

ax2.set_title('Daily increase',size=16);