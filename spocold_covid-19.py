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

            ax.text(day-0.5, scale * y, text,

                verticalalignment='bottom', 

                horizontalalignment='right',

                color='blue', fontsize=10)   

    plt.show()

    

    return

print('Current data set is "covid_19_clean_complete.csv" from kaggle.com by Devakumar KP')

print('data from ',df[:1]['Date'].to_string(index=False),' till ', df[-1:]['Date'].to_string(index=False))
from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
# country = 'Norway'

country = 'Netherlands'

rdf = df[df['Country/Region']==country]

rdf

# create_download_link(region_df)
# create_download_link(rdf)

rdf2 = rdf.iloc[::2]

netherlands = rdf2.iloc[::2]   # Убрал регионы, оставил только данные по стране полностью

netherlands
show_est(rdf,[[36, 0.2, 'the first confirmed case of COVID-19 (27 Feb)', 'black'],

              [44, 0.25, 'the RIVM announced the first death due to COVID-19, an 86-year-old patient (6 March)', 'black'],

              [49, 0.50, 'Larger events were banned, including football, festivals, parades and concerts (11 March)', 'black'],

              [50, 0.55, 'All events >100 people are now forbidden. Universities suspended, schools remain open (12 March)', 'black'],

              [51, 0.60, 'cancelled all flights from China, Iran, Italy, and South Korea, for two weeks (13 March)', 'black'],

              [53, 0.65, '1,135 cases. Schools, cafés, restaurants, sports clubs closed (15 March)', 'black'],

              [56, 0.70, 'the minister for Medical care got sick during the debate and collapsed (18 March)', 'black']

              ], d_rate=10.0,smooth_koef=0.0,shift_deaths_day=1,shift_confirm_day=1)



# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_the_Netherlands