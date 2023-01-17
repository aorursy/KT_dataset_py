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
country = 'Spain'

region_df = df[df['Country/Region']==country]

 

show_est(region_df,[[9, 0.8, ' On 31 January 2020, Spain confirmed its first COVID-19 case in La Gomera, Canary Islands.', 'black'],

                    [46, 0.75, 'In La Rioja, Haro was put on lockdown 7 March', 'black'],

                    [48, 0.5,'the closure of all schools in the municipalities of Vitoria and Labastida 9 March', 'black'],

                    [48, 0.3,'the cancellation of classes in the Autonomous community of Madrid at all educational levels 9 March', 'black'],

                    [49, 0.4,'events with more than one thousand attendants in Madrid, La Rioja and Vitoria were suspended 10 March', 'black'],

                    [51, 0.55,'Nationwide closure of schools 12 March', 'black'],

                    [53, 0.6,'The Spanish government imposes a nationwide lockdown 14 March', 'black'],

                    [54, 0.7,'a State of Alarm and announced the imposition of a national lockdown 15 March', 'black'],

                    [55, 0.8,'the closing of Spanish frontiers 16 March', 'black'],

                    

                    

                   ],d_rate=10.0,smooth_koef=0.0,shift_deaths_day=18,shift_confirm_day=6)

#plt.yscale('log')





# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Italy