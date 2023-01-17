# run each cell by Shift+Enter

# daily_infected returns the number of infected in a given region under the assumptions 1-3. 



dayshift = 16



def estimation(timeseries, multiplier=100.0/4.0, shift=dayshift):      

    timeseries = multiplier * timeseries

    timeseries[:-shift] = timeseries[shift:]

    timeseries[-shift:] = 0

    return timeseries



import pandas as pd    

import numpy as np    

import matplotlib.pyplot as plt

from IPython.display import Image

global ax



%matplotlib inline 

plt.rcParams['figure.figsize'] = 14, 10





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



def show_est(data,vert_lines,d_rate=4.0,smooth_koef=0.0,shift_deaths_day=17,shift_confirm_day=5): 

    plt.figure(figsize=(14, 14), dpi=50)

    confirmed_per_day = estimation(cum2daily(smooth(data['Confirmed'],smooth_koef)),

                                   multiplier=4.0,shift=shift_confirm_day)

    ax = show_with_dates(confirmed_per_day[:-shift_confirm_day],

         data['Date'],'Infected per day, estimation by confirmed').plot.line()



    infected_per_day = estimation(cum2daily(smooth(data['Deaths'],smooth_koef)),

                                  multiplier=100.00/d_rate,shift=shift_deaths_day)

    show_with_dates(infected_per_day[:-shift_deaths_day],

         data['Date'],'Infected per day, estimation by deaths').plot.line(ax=ax)



    scale = max(confirmed_per_day)

    ax.xaxis.set_ticks(range(0,len(data['Date'])))

    ax.set_xticklabels(list(data['Date']), rotation=90)

    

    for day, y, text, color in vert_lines:

            plt.axvline(x=day,color=color)

            ax.text(day+0.5, scale * y, text,

                verticalalignment='bottom', 

                horizontalalignment='left',

                color='blue', fontsize=10)   

    plt.show()

    

    return

print('Current data set is "covid_19_clean_complete.csv" from kaggle.com by Devakumar KP')

print('data from ',df[:1]['Date'].to_string(index=False),' till ', df[-1:]['Date'].to_string(index=False))
### 2.1 Greece
country = 'Greece'

region_df = df[df['Country/Region']==country]

 

show_est(region_df,[[9, 0.8, ' The coronavirus pandemic first appeared in Greece on 26 February 2020', 'black'],

                    [46, 0.75, 'On 9 March, all school trips were banned, all school championships were cancelled', 'black'],

                    [48, 0.5,'Starting on 10 March, all educational institutions were closed for 14 days', 'black'],

                    [48, 0.3,'On 12 March, a two-week closure of all theatres, courthouses, cinemas, gyms, playgrounds and clubs was announced', 'black'],

                    [49, 0.4,'On 16 March, Greece closed its borders with Albania and North Macedonia, deciding to suspend all road, sea and air links with these countries', 'black'],

                    [51, 0.55,'On 18 March, Deputy Minister of Civil Protection and Crisis Management Nikos Hardalias announced a ban on public gatherings of 10 or more people and the imposition of a 1,000 euro fine on violators', 'black'],

                    [53, 0.6,'On 19 March, the government announced the closure of all hotels across the country, from midnight on March 22 and until the end of April', 'black'],

                    [54, 0.7,'On 20 March, only permanent residents and supply trucks would be allowed to travel to the Greek islands, with effect from 6am local time on 21 March', 'black'],

                    [55, 0.8,'On 22 March, a ban on all nonessential transport and movement across the country, starting from 6 a.m on 23 March until 6 April', 'black'],

                                       

                   ],d_rate=10.0,smooth_koef=0.0,shift_deaths_day=18,shift_confirm_day=6)

#plt.yscale('log')





# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Greece