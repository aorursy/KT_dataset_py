#!pip install reverse_geocoder
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import reverse_geocoder as rg ## Gives country/county information based on zipcode

import matplotlib.pyplot as plt ## Plotting

import matplotlib.ticker as tck ## modify ticks on plot

import datetime ## parse date and time type
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'

df = pd.read_csv(url)
#county = 'Iredell'

#state = 'North Carolina'



county = 'Onondaga'

state = 'New York'
df_state_raw = df.loc[ (df['Province_State'] == state)]

df_cty = df_state_raw.loc[(df['Admin2'] == county)]
# Drop the columns which do not correspond to dates

df_cty.drop(columns=['UID', 'iso2','iso3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_','Combined_Key','code3'],inplace=True)

df_state_raw.drop(columns=['UID', 'iso2','iso3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_','Combined_Key','code3'],inplace=True)



# Reformat date strings

dates = [col for col in df_cty.columns]



for col in dates:

    split_date = col.split('/')

    new_date = "".join([split_date[0],'/',split_date[1]])

    

    df_cty.rename(columns={col:new_date},inplace=True)

    df_state_raw.rename(columns={col:new_date},inplace=True)

    

# Sum up that state data

df_state = df_state_raw.sum()



# Convert the dataframe of the county into a data series

df_cty = df_cty.T.squeeze()



# Drop dates which do not have any cases

dates = []

dates = [col for col in df_cty.index]



for col in dates:

    if df_cty[col] == 0:

        df_cty.drop(index=[col],inplace = True)

    if df_state[col] == 0:

        df_state.drop(index=[col],inplace = True)

## delay the proper start till n number of cases ##

n_state = 100

n_cty = 10



df_state_n_cases = df_state[df_state.values>n_state]

df_cty_n_cases = df_cty[df_cty.values>n_cty]





first_case_state = df_state_n_cases.iloc[0]

first_case_cty = df_cty_n_cases.iloc[0]



## two day doubling

two_day_double_state = [first_case_state, first_case_state * (2)**(len(df_state_n_cases)//2)]

two_day_double_cty =  [first_case_cty, first_case_cty * (2)**((len(df_cty_n_cases)//2))]



## three day doubling

three_day_double_state = [first_case_state, first_case_state * (2)**(len(df_state_n_cases)//3) ]

three_day_double_cty =   [first_case_cty,   first_case_cty * (2)**((len(df_cty_n_cases)//3))   ]



## First and last date of reporting

first_last_day_state = [df_state_n_cases.index[0],df_state_n_cases.index[-1]]    

first_last_day_cty = [df_cty_n_cases.index[0],df_cty_n_cases.index[-1]]



        
plot_deaths = False



fig, ax = plt.subplots(figsize=(20,10))



############

## Cases

############



## marker size

ms = 10



## State Cases and Deaths

cases_state, = plt.plot(df_state.index,df_state.values,

                        marker='o',markersize=ms, color='blue', 

                        label = 'Cases (State)')

if plot_deaths:

    deaths_state, = plt.plot(df_state.index,df_state.values,

                             marker='d',markersize=ms,color='blue',

                             label = 'Deaths (State)')





## County Cases and Deaths

cases_cty, = plt.plot(df_cty.index,df_cty.values,

                      marker='o',markersize=ms, color='red',

                      label = 'Cases (County)')

if plot_deaths:

    deaths_cty, = plt.plot(df_cty.index,df_cty.values,

                           marker='d',markersize=ms,color = 'red',

                           label = 'Deaths (County)')



############

## Bounds

############



## Line width

lw = 4





## State Bounds

two_day_state,= plt.plot(first_last_day_state,two_day_double_state,

                         ls='--',markersize=ms, color='blue',linewidth = lw,

                         label = 'Two day doubling')

three_day_state,= plt.plot(first_last_day_state,three_day_double_state,

                           ls=':', markersize=ms, color='blue', linewidth = lw,

                           label = 'Three day doubling')



## County bounds

two_day_cty,= plt.plot(first_last_day_cty,two_day_double_cty,

                       ls='--',  color='red',linewidth = lw,

                       label = 'Two day doubling')

three_day_cty,= plt.plot(first_last_day_cty,three_day_double_cty,

                         ls=':',  color='red', linewidth = lw,

                         label = 'Three day doubling')









############

## Plot Aesthetics

############

plt.yscale('log')

ax.get_yaxis().set_major_formatter(tck.ScalarFormatter())

ax.yaxis.set_tick_params(labelsize=20)

ax.yaxis.grid(b=True,which='major')

plt.ylabel('Number of Confirmed Cases',fontsize=18)

plt.ylim((1, 100000))



ax.xaxis.set_tick_params(labelsize=20)

plt.xticks(rotation=45) 

plt.xlabel('Date',fontsize=18)



############

## Legends and Title

############



##Font size

fs = 18



case_legend = plt.legend(handles=[cases_state,cases_cty], 

                         loc=(.008,.877),fontsize=fs,markerfirst = False)

for handle in case_legend.legendHandles: handle.set_linestyle("")

plt.gca().add_artist(case_legend)                     

                     

bound_legend = plt.legend(handles=[two_day_cty,three_day_cty], 

                          loc=(0.008,.7),fontsize=fs,markerfirst = False)

for handle in bound_legend.legendHandles:  handle.set_color('black') 

plt.gca().add_artist(bound_legend) 





if plot_deaths:

    death_legend = plt.legend(handles=[deaths_state,deaths_cty], 

                              loc=(0.2,.877),fontsize=fs,markerfirst = False)

    for handle in case_legend.legendHandles: handle.set_linestyle("")

    plt.gca().add_artist(case_legend)



plt.title('COVID-19 in %s State / %s County' %(state,county),fontsize=fs)







plt.savefig('covid_in_%s_county.png'%(county),dpi=300)
