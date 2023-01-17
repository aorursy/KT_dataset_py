# run each cell by Shift+Enter



# assumption: assumed that the mortal rate in a given region is a constant

# the function daily_infected returns the number of infected in a given region under the assumption 

def daily_infected(timeseries, multiplier=100.0/4.0, shift=20):      # Baud D. et al. Real estimates of mortality following COVID-19 infection //The Lancet Infectious Diseases. – 2020. https://doi.org/10.1016/S1473-3099(20)30195-X

    timeseries = multiplier * timeseries

    timeseries[:-shift] = timeseries[shift:]

    timeseries[-shift:] = 0

    return timeseries



global ax





import pandas as pd   #Библиотека для работы с базами данных

import numpy as np    #Библиотека для работы с массивами числовых данных

import matplotlib.pyplot as plt

from IPython.display import Image

%matplotlib inline    



df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv') 



countries = list(df['Country/Region'].unique())

def smooth(time_series,eps=0.2):    # this function smooths timeseries by three elements

    timeseries = np.zeros(len(time_series)+2)

    timeseries[1:-1] = np.array(time_series)

    timeseries[0], timeseries[-1] = timeseries[1], timeseries[-2]

    return pd.Series(eps*timeseries[:-2]+(1-2*eps)*timeseries[1:-1]+eps*timeseries[2:])



def smooth5(timeseries,eps=0.15):   # this function smooths timeseries by five elements

    timeseries = np.zeros(len(time_series)+4)

    timeseries[2:-2] = np.array(time_series)

    timeseries[0], timeseries[1], timeseries[-2], timeseries[-1] = timeseries[2], timeseries[2], timeseries[-3], timeseries[-3]

    return pd.Series(eps*timeseries[:-4]+(1/3-eps/3)*timeseries[1:-3]+(1/3-4*eps/3)*timeseries[2:-2]+(1/3-eps/3)*timeseries[3:-1]+eps*timeseries[4:])



def cum2daily(time_series):      # this function is an inverse of cummulative summation

    timeseries = np.zeros(len(time_series)+1)

    timeseries[1:] = np.array(time_series)

    timeseries[0] = timeseries[1]

    return pd.Series(timeseries[1:]-timeseries[:-1])



def show_with_dates(columns,dates, label):

    columns = pd.DataFrame(columns, columns=[label])

    columns['Date']=dates.reset_index()['Date'] 

    return columns.set_index('Date')



def show_est(data,vert_lines,death_rate=4.0,smooth_koef=0.15): 

    confirmed_per_day = cum2daily(smooth(data['Confirmed'],0.3))

    ax = show_with_dates(confirmed_per_day,

                   data['Date'],'Confirmed per day, smoothed').plot.line()

    infected_per_day = daily_infected(cum2daily(smooth(data['Deaths'],smooth_koef)),multiplier=100/death_rate,shift=20)

    show_with_dates(infected_per_day[:-20],

              data['Date'],'Infected per day, reconstruction').plot.line(ax=ax)

    

    scale = max(infected_per_day)

    

    for day, y, text, color in vert_lines:

              plt.axvline(x=day,color=color)

              ax.text(day+1, scale * y, text,

                        verticalalignment='bottom', horizontalalignment='left',

                        color=color, fontsize=10)   

print('The data set is actual on', df[-1:]['Date'].to_string(index=False))
country = 'Norway'

region_df = df[df['Country/Region']==country]

plt.figure()    

show_est(region_df,[[30, 0.5, '16 cases in Lombardy on 21 Feb', 'black'],

                    [0, 0.9, country, 'black'],

                    [len(region_df)-20, 0, 'data on last 20 days is N/A', 'orange']],death_rate=4.0)

plt.show()
Image("/kaggle/input/weather2/Norway_oslo.png",width=432) # thanks https://meteostat.net/en/station
region_df = df[df['Country/Region']=='Sweden']



# img = plt.imread('/kaggle/input/weather2/Sweden_stockholm1.png')

# x = range(len(region_df))

# fig, ax = plt.subplots(dpi=400) 

# ax.imshow(img, extent=[0, len(region_df), 0, 10])

show_est(region_df,[[30, 0.5, '16 cases in Lombardy on 21 Feb', 'black'],

                    [0, 0.9, country, 'black'],

                    [len(region_df)-20, 0, 'data on last 20 days is N/A', 'orange']],death_rate=4.0)





plt.show()



Image("/kaggle/input/weather3/Sweden_stockholm1.png",width=382) # thanks https://meteostat.net/en/station

 