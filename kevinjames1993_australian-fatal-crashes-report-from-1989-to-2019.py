# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input/accidentfatal'):

 #   for filename in filenames:

        #print(os.path.join(dirname, filename))

#for dirname, _, filenames in os.walk('/kaggle/input/australianpopulation'):

    #for filename_1 in filenames:

        #print(os.path.join(dirname, filename_1))



# Any results you write to the current directory are saved as output.
#!/usr/bin/env python

# coding: utf-8



# <h1> ICT Project 1



# <h2> Australian Road Deaths Database (ARDD) 

# <h3>

#     Source: <a href="https://data.gov.au/dataset/ds-dga-5b530fb8-526e-4fbf-b0f6-aa24e84e4277/details?q=road%20crash"> Australia Gov.data.au</a>



# In[1]:





import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import folium

import  scipy.interpolate as ip



def state_full_name(state):

    state = state.upper()

    if state == 'NSW':

        return 'New South Wales'

    elif state == 'QLD':

        return 'Queensland'

    elif state == 'VICTORIA' or state == 'VIC':

        return 'Victoria'

    elif state == 'ACT':

        return 'Australian Capital Territory'

    elif state == 'NT':

        return 'Northern Territory'

    elif state == 'SA':

        return 'South Australia'

    elif state == 'WA':

        return 'Western Australia'

    elif state == 'TAS':

        return 'Tasmania'

    

def speed_group(speed):        

    if speed <75:

        return '0 - 75'

    elif speed >=75:

        return 'Above 75'

    

def age_group(age):

    if age <= 18:

        return '0 to 18'

    elif age <= 30:

        return '19 to 30'

    elif age <=50:

        return '31 to 50'

    elif age >50:

        return '51 & Older'

def time_frame(time):

#     print(time)

    if time == '-9':

        return '6 PM - Midnight'

    elif int(time[0:-3]) <6:

        return 'Midnight - 6 AM'

    elif int(time[0:-3]) <12:

        return '6 AM -12 PM'

    elif int(time[0:-3]) <18:

        return '12 PM - 6 PM'

    else:

        return '6 PM - Midnight'



# time = "-9"





def get_month_number(month):

    return str(month)[5:7]



def get_year_number(year):

    return int(str(year)[0:4])



aus_geo = r'https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.geojson'

# create a plain world map

latitude = -25

longitude = 125



df_fatalities_data = pd.read_csv('/kaggle/input/accidentfatal/ardd_fatalities.csv', delimiter=',')

df_fatal_data = pd.read_csv('/kaggle/input/accidentfatal/ardd_fatal_crashes.csv', delimiter=',')

# url_population = 'https://www.abs.gov.au/AUSSTATS/ABS@Archive.nsf/log?openagent&310104.xls&3101.0&Time%20Series%20Spreadsheet&D5044B5F65AFD893CA25852F001DE5F5&0&Sep%202019&19.03.2020&Latest'

# df_pop = pd.read_excel(url_population, sheet_name='Data1')

df_pop=pd.read_excel('/kaggle/input/australianpopulation/australianpopulation.xls', sheet_name='Data1')



#df_fatalities_data.dataframeName = 'ardd_fatalities.csv'

#df_fatal_data.dataframeName = 'ardd_fatal_crashes.csv'

df_fatal_data.rename(columns={'Bus \nInvolvement': 'Bus Involvement'}, inplace=True)

df_fatal_data.rename(columns={'Bus \nInvolvement': 'Bus Involvement'}, inplace=True)



# Data Preprocessing

df_fatal_data = df_fatal_data[df_fatal_data['Speed Limit'] != '-9']

df_fatal_data = df_fatal_data[df_fatal_data['Bus Involvement'] != '-9']

df_fatalities_data = df_fatalities_data[df_fatalities_data['Gender'] != '-9']

df_fatalities_data = df_fatalities_data[df_fatalities_data['Bus Involvement'] != '-9']

df_fatal_data['Speed Limit'].replace(['<40'],'39',inplace=True)

df_fatal_data['Speed Limit'].replace(['Unspecified'],'0',inplace=True)

df_fatal_data['Speed Limit'] = pd.to_numeric(df_fatal_data['Speed Limit'])



df_fatalities_data = df_fatalities_data[df_fatalities_data['Speed Limit'] != '-9']

df_fatalities_data['Speed Limit'].replace(['<40'],'39',inplace=True)

df_fatalities_data['Speed Limit'].replace(['Unspecified'],'0',inplace=True)

df_fatalities_data['Speed Limit'] = pd.to_numeric(df_fatalities_data['Speed Limit'])



# Dataset filtered with Year > 2014

df_fatal_data_filtered_1 = df_fatal_data[df_fatal_data['Year']>1988]

df_fatal_data_filtered = df_fatal_data_filtered_1[df_fatal_data_filtered_1['Year']<2020]

df_fatalities_data_filtered_2 = df_fatalities_data[df_fatalities_data['Year']>1988]

df_fatalities_data_filtered = df_fatalities_data_filtered_2[df_fatalities_data_filtered_2['Year']<2020]



df_pop = df_pop[df_pop.columns.drop(list(df_pop.filter(regex='ale ;')))]

df_pop.rename(columns={'Unnamed: 0': 'Year_Month_Date', 'Estimated Resident Population ;  Persons ;  New South Wales ;':'NSW',

'Estimated Resident Population ;  Persons ;  Victoria ;':'Vic',

'Estimated Resident Population ;  Persons ;  Queensland ;':'Qld',

'Estimated Resident Population ;  Persons ;  South Australia ;':'SA',

'Estimated Resident Population ;  Persons ;  Western Australia ;':'WA',

'Estimated Resident Population ;  Persons ;  Tasmania ;':'Tas',

'Estimated Resident Population ;  Persons ;  Northern Territory ;':'NT',

'Estimated Resident Population ;  Persons ;  Australian Capital Territory ;':'ACT',

'Estimated Resident Population ;  Persons ;  Australia ;':'Australia'

}, inplace=True)

df_pop.drop(df_pop.index[0:9], inplace=True)

df_pop['Month_number'] = df_pop['Year_Month_Date'].apply(get_month_number)

df_pop=df_pop[df_pop['Month_number']=='06']

df_pop['Year'] = df_pop['Year_Month_Date'].apply(get_year_number)
def time_frame_hour(time):

#     print(time)

    if time == '-9':

        return '23'

    elif int(time[0:-3]) <1:

        return '00'

    elif int(time[0:-3]) <2 > 1:

        return '01'

    elif int(time[0:-3]) <3>2:

        return '02'

    elif int(time[0:-3]) <4>3:

        return '03'

    elif int(time[0:-3]) <5>4:

        return '04'

    elif int(time[0:-3]) <6>5:

        return '05'

    elif int(time[0:-3]) <7>6:

        return '06'

    elif int(time[0:-3]) <8>7:

        return '07'

    elif int(time[0:-3]) <9>8:

        return '08'

    elif int(time[0:-3]) <10>9:

        return '09'

    elif int(time[0:-3]) <11>10:

        return '10'

    elif int(time[0:-3]) <12>11:

        return '11'

    elif int(time[0:-3]) <13>12:

        return '12'

    elif int(time[0:-3]) <14>13:

        return '13'

    elif int(time[0:-3]) <15>14:

        return '14'

    elif int(time[0:-3]) <16>15:

        return '15'

    elif int(time[0:-3]) <17>16:

        return '16'

    elif int(time[0:-3]) <18>17:

        return '17'

    elif int(time[0:-3]) <19>18:

        return '18'

    elif int(time[0:-3]) <20>19:

        return '19'

    elif int(time[0:-3]) <21>20:

        return '20'

    elif int(time[0:-3]) <22>21:

        return '21'

    elif int(time[0:-3]) <23>22:

        return '22'

    else:

        return '23'
#fatality rate per 100,000 people

df_fatality_by_year = df_fatal_data_filtered.groupby('Year')['Number Fatalities'].sum()

df_fatality_by_year = df_fatality_by_year.to_frame().reset_index()

df_fatality_by_year_population = df_fatality_by_year.merge(df_pop, how='inner', left_on='Year', right_on='Year')

df_fatality_by_year_population['Number_of_fatality'] = 100000*df_fatality_by_year_population['Number Fatalities']/df_fatality_by_year_population['Australia']

df_fatality_by_year_1m = df_fatality_by_year_population[['Year', 'Number_of_fatality']]

df_fatality_by_year_1m.set_index('Year', inplace=True)

#plot graph

df2 = df_fatality_by_year_1m.reset_index()

data_x = np.array( df2['Year'])

data_y1 = np.array(df2['Number_of_fatality'] )



t1, c1, k1 = ip.splrep(data_x, data_y1, s=0, k=4)



data_x_smooth = np.linspace(data_x.min(), data_x.max(), 100)

data_y1_smooth = ip.BSpline(t1,c1,k1, extrapolate=False)



plt.figure(figsize=(16,8), linewidth=4)

plt.plot(data_x_smooth, data_y1_smooth(data_x_smooth), 'r', label='Fatalities')

rolling_mean = df2['Number_of_fatality'].rolling(window=3).mean()

plt.plot(data_x, rolling_mean, 'b',label= 'Average')

plt.plot(data_x, data_y1, 'rD',linestyle ='none')



plt.xlabel("Year", fontsize=14)

plt.ylabel("Fatalities", fontsize=14)

plt.title("Fatality per year per 100,000 people", fontsize=16)

plt.legend(loc='best')

plt.grid()

plt.show()
df_fatality_by_states = df_fatal_data_filtered.groupby(['State', 'Year'],  as_index=False)['Number Fatalities'].sum()

df_pop_statewise = df_pop[['NSW','Vic','Qld','SA','WA','Tas','NT','ACT','Australia','Year']]

df_pop_statewise = df_pop_statewise.melt(id_vars = 'Year', var_name='state', value_name='population')

df_fatality_by_states  = df_fatality_by_states.merge(df_pop_statewise, how='left', left_on=['Year','State'], right_on=['Year','state'])

df_fatality_by_states['Fatality'] = 100000 *  df_fatality_by_states['Number Fatalities']/df_fatality_by_states['population']

df_fatality_by_states = df_fatality_by_states[['State', 'Year', 'Fatality']]



df_fatality_by_states = pd.pivot_table(data=df_fatality_by_states, values='Fatality', index=['Year'], columns=['State'], aggfunc='sum')

df_fatality_by_states['ACT'].fillna(0, inplace=True)



col_list = df_fatality_by_states.columns

df_list = []

for col in col_list:

    df2 = df_fatality_by_states[col].reset_index()

    data_x = np.array( df2['Year'])

    data_y1 = np.array(df2[col])

    

    t1, c1, k1 = ip.splrep(data_x, data_y1, s=0, k=4)

    

    data_x_smooth = np.linspace(data_x.min(), data_x.max(), 250)

    data_y1_smooth = ip.BSpline(t1,c1,k1, extrapolate=False)

    y_data =  data_y1_smooth(data_x_smooth)

    

    df = pd.DataFrame(y_data, data_x_smooth)

    df_list.append(df)



nrow=2

ncol=4



fig, axes = plt.subplots(nrow, ncol, figsize=(20,10))

count=0



for r in range(nrow):

    for c in range(ncol):

        if count < len(df_list):

            axes[r, c].plot(df_fatality_by_states[col_list[count]],'-bD', linestyle = 'None')

            axes[r, c].plot(df_list[count])

            axes[r, c].set_title(col_list[count].upper())

            

        else:

            axes[r, c].set_visible(False)

        count+=1



fig.suptitle('Fatality by State per year per 100,000 people')



for ax in axes.flat:

    ax.set(ylabel='Fatalies', xlabel='Year')
# Fatality rate by Crash Type

piv_fatality = pd.pivot_table(data=df_fatal_data_filtered, values='Number Fatalities', index=['Year'], columns=['Crash Type'], aggfunc='sum')

piv_fatality['Total Fatality'] = piv_fatality['Multiple'] + piv_fatality['Pedestrian'] + piv_fatality['Single']

piv_fatality['Multiple'] = round(100 *piv_fatality['Multiple']/piv_fatality['Total Fatality'],2)

piv_fatality['Pedestrian'] = round(100 *piv_fatality['Pedestrian']/piv_fatality['Total Fatality'],2)

piv_fatality['Single'] = round(100 *piv_fatality['Single']/piv_fatality['Total Fatality'],2)

mask = ['Single','Multiple', 'Pedestrian']

piv_fatality = piv_fatality[mask]



#plotting graph



df2 = piv_fatality[:]

df2 = df2.reset_index()

data_x = np.array( df2['Year'])

data_y1 = np.array(df2['Single'] )

data_y2 = np.array(df2['Multiple'])

data_y3 = np.array(df2['Pedestrian'])



t1, c1, k1 = ip.splrep(data_x, data_y1, s=0, k=4)

t2, c2, k2 = ip.splrep(data_x, data_y2, s=0, k=4)

t3, c3, k3 = ip.splrep(data_x, data_y3, s=0, k=4)



data_x_smooth = np.linspace(data_x.min(), data_x.max(), 150)

data_y1_smooth = ip.BSpline(t1,c1,k1, extrapolate=False)

data_y2_smooth = ip.BSpline(t2,c2,k2, extrapolate=False)

data_y3_smooth = ip.BSpline(t3,c3,k3, extrapolate=False)



plt.figure(figsize=(16,8), linewidth=4)

plt.plot(data_x_smooth, data_y1_smooth(data_x_smooth), 'r', label='Single')

plt.plot(data_x_smooth, data_y2_smooth(data_x_smooth), 'b', label='Multiple')

plt.plot(data_x_smooth, data_y3_smooth(data_x_smooth), 'g', label='Pedestrian')

plt.plot(data_x, data_y1, 'rD',linestyle ='none')

plt.plot(data_x, data_y2, 'bD',linestyle ='none')

plt.plot(data_x, data_y3, 'gD',linestyle ='none')

plt.xlabel("Year", fontsize=14)

plt.ylabel("Fatality Rate", fontsize=14)

plt.ylim(10,55)

plt.title("Fatality Rate by Crash Type per year", fontsize=16)

plt.legend(loc='upper right')

plt.grid()

plt.show()
#Road user Fatality Rate

piv_fatility_Roaduser = pd.pivot_table(data=df_fatalities_data_filtered,values='Crash ID', index=['Year'], columns=['Road User'], aggfunc='count')

piv_fatility_Roaduser['Others'] =piv_fatility_Roaduser['Motorcycle pillion passenger']+ piv_fatility_Roaduser['Other/-9']

piv_fatility_Roaduser['Total Fatality'] = piv_fatility_Roaduser['Driver']+ piv_fatility_Roaduser['Motorcycle pillion passenger']+piv_fatility_Roaduser['Motorcycle rider']+piv_fatility_Roaduser['Passenger']+piv_fatility_Roaduser['Pedal cyclist']+piv_fatility_Roaduser['Pedestrian']

piv_fatility_Roaduser['Driver'] = round(100 *piv_fatility_Roaduser['Driver']/piv_fatility_Roaduser['Total Fatality'],2)

piv_fatility_Roaduser['Motorcycle pillion passenger'] = round(100 *piv_fatility_Roaduser['Motorcycle pillion passenger']/piv_fatility_Roaduser['Total Fatality'],2)

piv_fatility_Roaduser['Motorcycle rider'] = round(100 *piv_fatility_Roaduser['Motorcycle rider']/piv_fatility_Roaduser['Total Fatality'],2)

piv_fatility_Roaduser['Passenger'] = round(100 *piv_fatility_Roaduser['Passenger']/piv_fatility_Roaduser['Total Fatality'],2)

piv_fatility_Roaduser['Pedal cyclist'] = round(100 *piv_fatility_Roaduser['Pedal cyclist']/piv_fatility_Roaduser['Total Fatality'],2)

piv_fatility_Roaduser['Pedestrian'] = round(100 *piv_fatility_Roaduser['Pedestrian']/piv_fatility_Roaduser['Total Fatality'],2)

piv_fatility_Roaduser['Others'] = round(100 *piv_fatility_Roaduser['Others']/piv_fatility_Roaduser['Total Fatality'],2)

mask = ['Driver','Motorcycle pillion passenger', 'Motorcycle rider','Passenger','Pedal cyclist','Pedestrian']

piv_fatility_Roaduser = piv_fatility_Roaduser[mask]



#Plot graph

col_list = piv_fatility_Roaduser.columns

df_list = []

for col in col_list:   

    df2 = piv_fatility_Roaduser[col].reset_index()

    data_x = np.array( df2['Year'])

    data_y1 = np.array(df2[col])

    

    t1, c1, k1 = ip.splrep(data_x, data_y1, s=0, k=4)

    

    data_x_smooth = np.linspace(data_x.min(), data_x.max(), 250)

    data_y1_smooth = ip.BSpline(t1,c1,k1, extrapolate=False)

    y_data =  data_y1_smooth(data_x_smooth)

    

    df = pd.DataFrame(y_data, data_x_smooth)

    df_list.append(df)





nrow=2

ncol=4



fig, axes = plt.subplots(nrow, ncol, figsize=(20,10))



count=0

for r in range(nrow):

    for c in range(ncol):

        if count < len(df_list):

            axes[r, c].plot(piv_fatility_Roaduser[col_list[count]],'-bD', linestyle = 'None')

            axes[r, c].plot(df_list[count])

            axes[r, c].set_title(col_list[count])            

        else:

            axes[r, c].set_visible(False)

        count+=1



fig.suptitle('Fatality Rate by Road User')



for ax in axes.flat:

    ax.set(ylabel='Fatality Rate', xlabel='Year')



# for ax in axes.flat:

#    ax.label_outer()
#Fatality by speed Limit (2015-2020)

df_fatality_for_speed_limit = df_fatal_data_filtered.loc[:, ['Speed Limit','Year']]

df_fatality_for_speed_limit['Speed Limit Group'] = df_fatality_for_speed_limit['Speed Limit'].apply(speed_group)

df_fatality_for_speed_limit = pd.pivot_table(data=df_fatality_for_speed_limit,values='Speed Limit', index=['Year'], columns=['Speed Limit Group'], aggfunc='count')

df_fatality_for_speed_limit['Total Fatalities']= df_fatality_for_speed_limit['0 - 75']+ df_fatality_for_speed_limit['Above 75']

df_fatality_for_speed_limit['0 - 75'] = round(100 * df_fatality_for_speed_limit['0 - 75']/df_fatality_for_speed_limit['Total Fatalities'],2)

df_fatality_for_speed_limit['Above 75'] = round(100 * df_fatality_for_speed_limit['Above 75']/df_fatality_for_speed_limit['Total Fatalities'],2)

mask = ['0 - 75','Above 75']

df_fatality_for_speed_limit = df_fatality_for_speed_limit[mask]



#plot graph

df2 = df_fatality_for_speed_limit[:]

df2 = df2.reset_index()

data_x = np.array( df2['Year'])

data_y1 = np.array(df2['0 - 75'] )

data_y2 = np.array(df2['Above 75'])



t1, c1, k1 = ip.splrep(data_x, data_y1, s=0, k=4)

t2, c2, k2 = ip.splrep(data_x, data_y2, s=0, k=4)



data_x_smooth = np.linspace(data_x.min(), data_x.max(), 150)

data_y1_smooth = ip.BSpline(t1,c1,k1, extrapolate=False)

data_y2_smooth = ip.BSpline(t2,c2,k2, extrapolate=False)



plt.figure(figsize=(16,8), linewidth=4)

plt.plot(data_x_smooth, data_y1_smooth(data_x_smooth), 'r', label='0 - 75')

plt.plot(data_x_smooth, data_y2_smooth(data_x_smooth), 'b', label='Above 75')

plt.plot(data_x, data_y1, '-rD', linestyle = 'None')

plt.plot(data_x, data_y2, '-bD', linestyle = 'None')



plt.xlabel("Year", fontsize=14)

plt.ylabel("Fatality Rate", fontsize=14)

plt.title("Fatality Rate by speed limit", fontsize=16)

plt.legend(loc='best')

plt.grid()

plt.show()
#fatality rate by Day/Night

piv_fatality_time_range = df_fatal_data_filtered.loc[:,['Year', 'Time', 'Number Fatalities']]

piv_fatality_time_range['time_range'] = piv_fatality_time_range['Time'].apply(time_frame)

piv_fatality_time_range = pd.pivot_table(data=piv_fatality_time_range, values='Number Fatalities',index=['Year'], columns=['time_range'], aggfunc='sum')

# df_fatal_data_filtered['Number Fatalities']

piv_fatality_time_range['Total'] =piv_fatality_time_range['Midnight - 6 AM'] +piv_fatality_time_range['6 AM -12 PM'] +piv_fatality_time_range['12 PM - 6 PM'] +piv_fatality_time_range['6 PM - Midnight'] 

piv_fatality_time_range ['Midnight - 6 AM'] = round(100 *piv_fatality_time_range['Midnight - 6 AM']/piv_fatality_time_range['Total'],2)

piv_fatality_time_range ['6 AM -12 PM'] = round(100 *piv_fatality_time_range['6 AM -12 PM']/piv_fatality_time_range['Total'],2)

piv_fatality_time_range ['12 PM - 6 PM'] = round(100 *piv_fatality_time_range['12 PM - 6 PM']/piv_fatality_time_range['Total'],2)

piv_fatality_time_range ['6 PM - Midnight'] = round(100 *piv_fatality_time_range['6 PM - Midnight']/piv_fatality_time_range['Total'],2)

mask = ['Midnight - 6 AM','6 AM -12 PM', '12 PM - 6 PM','6 PM - Midnight']

piv_fatality_time_range = piv_fatality_time_range[mask]

piv_fatality_time_range.head(10)



#plot graph

col_list = piv_fatality_time_range.columns

df_list = []

for col in col_list:

    df2 = piv_fatality_time_range[col].reset_index()

    data_x = np.array( df2['Year'])

    data_y1 = np.array(df2[col])

    

    t1, c1, k1 = ip.splrep(data_x, data_y1, s=0, k=4)

    

    data_x_smooth = np.linspace(data_x.min(), data_x.max(), 250)

    data_y1_smooth = ip.BSpline(t1,c1,k1, extrapolate=False)

    y_data =  data_y1_smooth(data_x_smooth)

    

    df = pd.DataFrame(y_data, data_x_smooth)

    df_list.append(df)

    



nrow=2

ncol=2



fig, axes = plt.subplots(nrow, ncol, figsize=(20,10))



count=0



for r in range(nrow):

    for c in range(ncol):

        if count < len(df_list):

            axes[r, c].plot(piv_fatality_time_range[col_list[count]],'-bD', linestyle = 'None')

            axes[r, c].plot(df_list[count])

            axes[r, c].set_title(col_list[count])

        else:

            axes[r, c].set_visible(False)

        count+=1



fig.suptitle('Fatality Rate by time range')



for ax in axes.flat:

    ax.set(ylabel='Fatality Rate', xlabel='Year')

piv_fatality_time_range_hour = df_fatal_data_filtered.loc[:,['Year', 'Time', 'Number Fatalities']]

piv_fatality_time_range_hour['time_range'] = piv_fatality_time_range_hour['Time'].apply(time_frame_hour)

piv_fatality_time_range_hour = pd.pivot_table(data=piv_fatality_time_range_hour, values='Number Fatalities',index=['time_range'], aggfunc='sum')

piv_fatality_time_range_hour.head(10)

ax = piv_fatality_time_range_hour.plot(

    kind='bar', 

    color = ('lightblue'),

    figsize=(13, 6) ,

    ylim=(1000,4000),

    width=1

)



ax.set_title("Fatality per hour", fontsize=16)

#ax.legend(loc='upper right', frameon=True, fontsize=14, bbox_to_anchor=(1.0, -0.1), ncol=5)

ax.set_ylabel("Fatalities", fontsize=14)

ax.set_xlabel("Time (00:00 to 23:00)", fontsize=12)

rolling_mean = piv_fatality_time_range_hour['Number Fatalities'].rolling(window=3).mean()

plt.plot( rolling_mean, 'r',label= 'Average')

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

plt.xticks(rotation=0)

plt.legend()

plt.grid(b=True, linewidth=0, axis='y');

rects = ax.patches
# Fatality Rate by day of weeek per year

piv_fatality_by_year_day = pd.pivot_table(data=df_fatal_data_filtered, values='Number Fatalities', index=['Year'], columns=['Dayweek'], aggfunc='sum')

piv_fatality_by_year_day['Total']=piv_fatality_by_year_day['Monday']+piv_fatality_by_year_day['Tuesday']+piv_fatality_by_year_day['Wednesday']+ piv_fatality_by_year_day['Thursday']+piv_fatality_by_year_day['Friday']+ piv_fatality_by_year_day['Saturday'] + piv_fatality_by_year_day['Sunday']

piv_fatality_by_year_day['Monday'] =  round(100 *piv_fatality_by_year_day['Monday']/piv_fatality_by_year_day['Total'],2)

piv_fatality_by_year_day['Tuesday']= round(100 *piv_fatality_by_year_day['Tuesday']/piv_fatality_by_year_day['Total'],2)

piv_fatality_by_year_day['Wednesday']= round(100 *piv_fatality_by_year_day['Wednesday']/piv_fatality_by_year_day['Total'],2)

piv_fatality_by_year_day['Thursday']= round(100 *piv_fatality_by_year_day['Thursday']/piv_fatality_by_year_day['Total'],2)

piv_fatality_by_year_day['Friday']= round(100 *piv_fatality_by_year_day['Friday']/piv_fatality_by_year_day['Total'],2)

piv_fatality_by_year_day['Saturday']= round(100 *piv_fatality_by_year_day['Saturday']/piv_fatality_by_year_day['Total'],2)

piv_fatality_by_year_day['Sunday']= round(100 *piv_fatality_by_year_day['Sunday']/piv_fatality_by_year_day['Total'],2)

# piv_fatality_by_year_day.reset_index(inplace=True)

Day_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday']

# piv_fatality_by_year_day['Dayweek'] = pd.Categorical(piv_fatality_by_year_day['Dayweek'], Day_week)

#piv_fatality_by_year_day = piv_fatality_by_year_day[mask]

piv_fatality_by_year_day = piv_fatality_by_year_day[Day_week]



#piv_fatality_by_state_day = piv_fatality_by_state_day[mask]

col_list = piv_fatality_by_year_day.columns



df_list = []

for col in col_list:

    df2 = piv_fatality_by_year_day[col].reset_index()

    data_x = np.array( df2['Year'])

    data_y1 = np.array(df2[col])  

    t1, c1, k1 = ip.splrep(data_x, data_y1, s=0, k=4)

    data_x_smooth = np.linspace(data_x.min(), data_x.max(), 250)

    data_y1_smooth = ip.BSpline(t1,c1,k1, extrapolate=False)

    y_data =  data_y1_smooth(data_x_smooth) 

    df = pd.DataFrame(y_data, data_x_smooth)

    df_list.append(df)



nrow=2

ncol=4



fig, axes = plt.subplots(nrow, ncol, figsize=(20,10))

# plot counter

count=0

# plt.figure(figsize=(16,8), linewidth=4)

# plt.figure(figsize=(40,20))

for r in range(nrow):

    for c in range(ncol):

        if count < len(df_list):

            axes[r, c].plot(piv_fatality_by_year_day[col_list[count]],'-bD', linestyle = 'None')

            axes[r, c].plot(df_list[count])

            axes[r, c].set_title(col_list[count])

            

        else:

            axes[r, c].set_visible(False)

        count+=1



fig.suptitle('Fatality by day of the week')

# plt.figure(figsize=(16,8))

for ax in axes.flat:

    ax.set(ylabel='Fatality', xlabel='Year')

# plt.figure(figsize=(40,20))

# plt.tight_layout()

# plt.show()
#Fatality Rate by Age group

df_fatalities_data_filtered = df_fatalities_data_filtered[df_fatalities_data_filtered['Age Group'] != '-9']

df_fatalities_data_filtered['Age_group_custom'] = df_fatalities_data_filtered['Age'].apply(age_group)

age_groups = ['0_to_16', '17_to_25', '26_to_39', '40_to_64', '65_to_74', '75_or_older']

df_fatality_by_age_group = df_fatalities_data_filtered.groupby(['Year','Age_group_custom'])['Age'].count()

df_fatality_by_age_year = df_fatalities_data_filtered.groupby(['Year'])['Crash ID'].count()



df_fatality_by_age_group = df_fatality_by_age_group.reset_index()

df_fatality_by_age_year = df_fatality_by_age_year.reset_index()

df_fatality_by_age_group = df_fatality_by_age_group.merge(df_fatality_by_age_year, how='inner', left_on='Year', right_on='Year')

df_fatality_by_age_group['Percent'] = round(100*df_fatality_by_age_group['Age']/df_fatality_by_age_group['Crash ID'], 2)

df_fatality_by_age_group.drop(['Age', 'Crash ID'], axis=1, inplace=True)

df_fatality_by_age_group = df_fatality_by_age_group.set_index(['Year', 'Age_group_custom'])['Percent'].unstack()



col_list = df_fatality_by_age_group.columns



df_list = []

for col in col_list:

    df2 = df_fatality_by_age_group[col].reset_index()

    data_x = np.array( df2['Year'])

    data_y1 = np.array(df2[col])

    

    t1, c1, k1 = ip.splrep(data_x, data_y1, s=0, k=4)

    

    data_x_smooth = np.linspace(data_x.min(), data_x.max(), 250)

    data_y1_smooth = ip.BSpline(t1,c1,k1, extrapolate=False)

    y_data =  data_y1_smooth(data_x_smooth)

    

    df = pd.DataFrame(y_data, data_x_smooth)

    df_list.append(df)



nrow=2

ncol=2



fig, axes = plt.subplots(nrow, ncol, figsize=(20,10))



count=0

for r in range(nrow):

    for c in range(ncol):

        if count < len(df_list):

            axes[r, c].plot(df_fatality_by_age_group[col_list[count]],'-bD', linestyle = 'None')

            axes[r, c].plot(df_list[count])

            axes[r, c].set_title(col_list[count])

            

        else:

            axes[r, c].set_visible(False)

        count+=1



fig.suptitle('Fatality Rate by age group')



for ax in axes.flat:

    ax.set(ylabel='Fatality Rate', xlabel='Year')
# Fatality Rate by Male from each state per year

df_fatility_male = df_fatalities_data_filtered[df_fatalities_data_filtered['Gender']=='Male']

piv_fatility_male = pd.pivot_table(data=df_fatility_male,values='Crash ID',index=['Year'], columns=['State'], aggfunc='count')

piv_fatility_male['ACT'].fillna(0, inplace=True)

piv_fatility_male['Total']=piv_fatility_male['ACT']+piv_fatility_male['NSW']+piv_fatility_male['NT']+piv_fatility_male['Qld']+piv_fatility_male['SA']+piv_fatility_male['Tas']+piv_fatility_male['Vic']+piv_fatility_male['WA']

mask = ['Total']

piv_fatility_male = piv_fatility_male[mask]

piv_fatility_male.head(10)

df2 = piv_fatility_male.reset_index()

data_x = np.array( df2['Year'])

data_y1 = np.array(df2['Total'] )



t1, c1, k1 = ip.splrep(data_x, data_y1, s=0, k=4)



data_x_smooth = np.linspace(data_x.min(), data_x.max(), 150)

data_y1_smooth = ip.BSpline(t1,c1,k1, extrapolate=False)



plt.figure(figsize=(16,8), linewidth=4)

plt.plot(data_x_smooth, data_y1_smooth(data_x_smooth), 'r', label='Fatalities')

rolling_mean = df2['Total'].rolling(window=3).mean()

plt.plot(data_x, rolling_mean, 'b',label= 'Average')

plt.plot(data_x, data_y1, 'rD',linestyle ='none')



plt.xlabel("Year", fontsize=14)

plt.ylabel("Fatalities", fontsize=14)

plt.title("Fatality by Male Per Year", fontsize=16)

plt.legend(loc='best')

plt.grid()

plt.show()
# Fatality Rate by Female of weeek per year

df_fatility_female = df_fatalities_data_filtered[df_fatalities_data_filtered['Gender']=='Female']

piv_fatility_female = pd.pivot_table(data=df_fatility_female,values='Crash ID',index=['Year'], columns=['State'], aggfunc='count')

piv_fatility_female['ACT'].fillna(0, inplace=True)

piv_fatility_female['Total']=piv_fatility_female['ACT']+piv_fatility_female['NSW']+piv_fatility_female['NT']+piv_fatility_female['Qld']+piv_fatility_female['SA']+piv_fatility_female['Tas']+piv_fatility_female['Vic']+piv_fatility_female['WA']

mask = ['Total']

piv_fatility_female = piv_fatility_female[mask]

piv_fatility_female.head(10)

df2 = piv_fatility_female.reset_index()

data_x = np.array( df2['Year'])

data_y1 = np.array(df2['Total'] )



t1, c1, k1 = ip.splrep(data_x, data_y1, s=0, k=4)



data_x_smooth = np.linspace(data_x.min(), data_x.max(), 150)

data_y1_smooth = ip.BSpline(t1,c1,k1, extrapolate=False)



plt.figure(figsize=(16,8), linewidth=4)

plt.plot(data_x_smooth, data_y1_smooth(data_x_smooth), 'r', label='Fatalities')

rolling_mean = df2['Total'].rolling(window=3).mean()

plt.plot(data_x, rolling_mean, 'b',label= 'Average')

plt.plot(data_x, data_y1, 'rD',linestyle ='none')



plt.xlabel("Year", fontsize=14)

plt.ylabel("Fatalities", fontsize=14)

plt.title("Fatality by Female Per Year", fontsize=16)

plt.legend(loc='best')

plt.grid()

plt.show()
