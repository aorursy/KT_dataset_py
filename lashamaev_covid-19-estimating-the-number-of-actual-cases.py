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



def smooth7(time_series,eps=0.1428):   # this function smooths timeseries by five elements

    timeseries = np.zeros(len(time_series)+6)

    timeseries[3:-3] = np.array(time_series)

    timeseries[0], timeseries[1], timeseries[2], timeseries[-3], timeseries[-2], timeseries[-1] = timeseries[3], timeseries[3], timeseries[3], timeseries[-4], timeseries[-4], timeseries[-4]

    return pd.Series((1-eps)*(timeseries[0:-6] + timeseries[1:-5]+\

        timeseries[2:-4]+timeseries[4:-2]+\

        timeseries[5:-1]+eps*timeseries[6:])/7+eps*timeseries[3:-3]/7)





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

    plt.figure(figsize=(14, 14), dpi=1)

    confirmed_per_day = estimation(cum2daily(smooth7(data['Confirmed'])),

                                   multiplier=4.0,shift=shift_confirm_day)

    ax = show_with_dates(confirmed_per_day[:-shift_confirm_day],

         data['Date'],'Symptomatic infections per day, estimation by confirmed').plot.line()



    infected_per_day = estimation(cum2daily(smooth7(data['Deaths'])),

                                  multiplier=100.00/d_rate,shift=shift_deaths_day)

    show_with_dates(infected_per_day[:-shift_deaths_day],

         data['Date'],'Symptomatic infections per day, estimation by deaths').plot.line(ax=ax)



    scale = max(confirmed_per_day)

    ax.xaxis.set_ticks(range(0,len(data['Date'])))

    ax.set_xticklabels(list(data['Date']), rotation=90)

    

    for day, y, text, color in vert_lines:

            plt.axvline(x=day,color=color)

            ax.text(day+0.5, scale * y, text,

                verticalalignment='bottom', 

                horizontalalignment='left',

                color='blue', fontsize=15)   

    plt.show()

    

    return

print('Current data set is "covid_19_clean_complete.csv" from kaggle.com by Devakumar KP')

print('data from ',df[:1]['Date'].to_string(index=False),' till ', df[-1:]['Date'].to_string(index=False))
country = 'Italy'

region_df = df[df['Country/Region']==country]

 

show_est(region_df,[[9, 0.8, ' flights to and from China suspended on 31 Jan', 'black'],

                    [30, 0.75, '16 cases in Lombardy on 21 Feb', 'black'],

                    [30, 0.7, 'lockdown in Lodi 21 Feb', 'black'],

                    [46, 0.65, 'lockdown in part of Northern Italy 8 March', 'black'],

                    [46, 0.6, 'laws for strict lockdown 8 March', 'black'],

                    [48, 0.55, 'National lockdown of Italy 10 March', 'black'],

                    [59, 0.5, 'strict lockdown on 21 March', 'black'],

                    [59, 0.45, 'almost all commer are shut down on 21 March', 'black']

                   ],d_rate=3.5,smooth_koef=0.15,shift_deaths_day=16,shift_confirm_day=12)

#plt.yscale('log')





# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Italy
country = 'France'

region_df = df[df['Country/Region']==country]

region_df = region_df[region_df['Province/State'].isna()]



show_est(region_df,[[9, 0.8, ' flights to and from China suspended on 31 Jan', 'black'],

                    [30, 0.75, '16 cases in Lombardy on 21 Feb', 'black'],

                    [54, 0.65, 'Schools and Universities are closed 16 March', 'black'],

                    [55, 0.55, 'Nationalwide lockdown of France 17 March', 'black'],

                   ],d_rate=3.5,smooth_koef=0.15,shift_deaths_day=18,shift_confirm_day=14)

#plt.yscale('log')





# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_France
country = 'United Kingdom'

region_df = df[df['Country/Region']==country]

region_df = region_df[region_df['Province/State'].isna()]



show_est(region_df,[[9, 0.8, ' flights to and from China suspended on 31 Jan', 'black'],

                    [30, 0.75, '16 cases in Lombardy on 21 Feb', 'black'],

                    [57, 0.55, 'Schools and Universities are closed 18 March', 'black'],

                    [58, 0.45, 'Restaurants, Pubs, Clubs are closed 20 March', 'black'],

                   ],d_rate=3.5,smooth_koef=0.0,shift_deaths_day=20,shift_confirm_day=18)

#plt.yscale('log')





# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_the_United_Kingdom
country = 'US'

region_df = df[df['Country/Region']==country]

# region_df['Province/State'].unique()

# region_df = region_df[region_df['Province/State']=='New York']



show_est(region_df,[[9, 0.8, ' flights to and from China suspended on 31 Jan', 'black'],

                    [30, 0.75, '16 cases in Lombardy on 21 Feb', 'black'],

                    [55, 0.45, 'Schools, bars, and restaurants in the NY are closed 17 March', 'black'],

                    [59, 0.65, 'Non-essential business are closed 21 March', 'black'],

                    [60, 0.55, 'Schools and Universities are closed 22 March', 'black'],                    

                   ],d_rate=1.5,smooth_koef=0.0,shift_deaths_day=20,shift_confirm_day=13)

#plt.yscale('log')





# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_the_US
 
country = 'Russia'

region_df = df[df['Country/Region']==country]

region_df = region_df[region_df['Province/State'].isna()]



 

 

show_est(region_df,[[67, 0.30, 'Putin started vacation days', 'black'],

                     ], d_rate=0.2,smooth_koef=0.0,shift_deaths_day=20,shift_confirm_day=19)
country = 'South Korea'

region_df = df[df['Country/Region']==country]

 

#filled by Qasqasqas    

show_est(region_df,[[27, 0.8, '31st case in Daegu on 18 Feb', 'black'],

                    [28, 0.75, '51 cases on 19 Feb', 'black'],

                    [29, 0.7, '104 cases on 20 Feb', 'black'],

                    [55, 0.55, 'around 79  RGC Church devotees infected on 17 March', 'black']

                   ],d_rate=0.3,smooth_koef=0.2,shift_deaths_day=20,shift_confirm_day=1)

# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_South_Korea


d_rate = {'Japan':0.5, 'Singapore':0.2, 'Norway':0.01, 'Sweden':3.0, 'Finland':1.0,'Taiwan*':3.0}

for country in d_rate.keys():

    region_df = df[df['Country/Region']==country]



    plt.figure()

    show_est(region_df,[[30, 0.5, '16 cases in Lombardy on 21 Feb', 'black'],

                        [0, 0.9, country, 'black'],

                        [len(region_df)-20, 0, 'data on last 20 days is N/A', 'orange']],d_rate[country],

                        smooth_koef=0.05,shift_deaths_day=19,shift_confirm_day=15)   

    plt.show()

country = 'Spain'

region_df = df[df['Country/Region']==country]

 

#filled by Nurgun

show_est(region_df,[[9, 0.8, 'the first case on 31 January 2020', 'black'],

                    [45, 0.75, 'Haro lockdown on 7 March', 'black'],

                    [47, 0.7,'schools in Vitoria and Labastida are closed on 9 March', 'black'],

                    [48, 0.65,'events in some cities with more 1000 attendants suspended on 10 March', 'black'],

                    [50, 0.6,'schools are closed on 12 March', 'black'],

                    [52, 0.55,'Nationwide lockdown on 14 March', 'black'],

                    [54, 0.5,'the closing of Spanish frontiers on 16 March', 'black'],

                   ],d_rate=3.0,smooth_koef=0.0,shift_deaths_day=17,shift_confirm_day=15)


d_rate = {'Iran':1.6}

for country in d_rate.keys():

    region_df = df[df['Country/Region']==country]



    plt.figure()

    show_est(region_df,[[30, 0.5, '16 cases in Lombardy on 21 Feb', 'black'],

                        [0, 0.9, country, 'black'],

                        [len(region_df)-20, 0, 'data on last 20 days is N/A', 'orange']],d_rate[country],

                        smooth_koef=0.0,shift_deaths_day=18,shift_confirm_day=14)

    plt.show()


d_rate = {'Greece':1.5, 'Portugal':1.0}

for country in d_rate.keys():

    region_df = df[df['Country/Region']==country]



    plt.figure()

    show_est(region_df,[[30, 0.5, '16 cases in Lombardy on 21 Feb', 'black'],

                        [0, 0.9, country, 'black'],

                        [len(region_df)-20, 0, 'data on last 20 days is N/A', 'orange']],d_rate[country],

                        smooth_koef=0.15,shift_deaths_day=20,shift_confirm_day=15)

    plt.show()
country = 'Ukraine'

region_df = df[df['Country/Region']==country]



#filled by Evgenia

show_est(region_df,[[9, 0.8, 'flights from Sanya suspended on 27 Jan', 'black'],

                    [30, 0.7, 'the first confirmed SARS-CoV-2 on 3 March', 'black'],

                    [46, 0.65, 'two more SARS-CoV-2 cases  in Ukraine 12 March', 'black'],

                    [46, 0.6, 'six more cases  in Chernivtsi on 17 March', 'black'],

                    [48, 0.55, 'the 3rd case in Kyiv on 19 March', 'black'],

                    [59, 0.5, 'the first case of recovery on 20 March', 'black'],

                    [59, 0.45, '26 new cases of COVID-19 were confimed on 22 March', 'black']

                   ],d_rate=0.6,smooth_koef=0.15,shift_deaths_day=18,shift_confirm_day=17)

 





# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Ukraine
country = 'Czechia'

region_df = df[df['Country/Region']==country]

 

# filled by Semen V.    

show_est(region_df,[[54, 0.8, 'Nationwide quarantine from 16 March', 'black'],

                   ],d_rate=0.8,smooth_koef=0.0,shift_deaths_day=20,shift_confirm_day=10)
country = 'Germany'

region_df = df[df['Country/Region']==country]



#filled by Ulyana

show_est(region_df,[[6, 0.65, 'Lufthansa suspended all flights to China on 28 Jan', 'black'],

                    [35, 0.6, 'North Rhine-Westphalia, Heinsberg closed schools, swimming pools, libraries and the town hall on 26 Feb', 'black'],

                    [37, 0.55, 'Heinsberg closed daycare facilities and schools on 28 Feb', 'black'],

                    [39, 0.5, 'The number of confirmed cases doubled on 1st March', 'black'],

                    [46, 0.45, 'Health Minister recommended cancelling 1000 attendees events on 8 March', 'black'],

                    [47, 0.4, 'Germany reported the first deaths on 9 March', 'black'],

                    [51, 0.35, '14 of the 16 German federal states closed schools and nurseries on 13 March', 'black'],

                      [58, 0.3, 'a curfew in Bavaria on 20 March', 'black'],

                    [60, 0.25, 'no more than two people gathering on 22nd March', 'black'],

                    [61, 0.2, 'financial aid package 750 billion on 23rd March', 'black'],

                   ],d_rate=1.0,smooth_koef=0.0,shift_deaths_day=23,shift_confirm_day=8)

region_df.columns
country = 'Netherlands'

region_df = df[df['Country/Region']==country]

region_df = region_df[region_df['Province/State'].isna()]



  



#filled by Aytal

show_est(region_df,[[36, 0.7, 'the first confirmed case of COVID-19 (27 Feb)', 'black'],

                     [44, 0.65, 'the RIVM announced the first death due to COVID-19, an 86-year-old patient (6 March)', 'black'],

                     [49, 0.60, 'Larger events were banned, including football, festivals, parades and concerts (11 March)', 'black'],

                     [50, 0.55, 'All events >100 people are now forbidden. Universities suspended, schools remain open (12 March)', 'black'],

                     [51, 0.50, 'cancelled all flights from China, Iran, Italy, and South Korea, for two weeks (13 March)', 'black'],

                     [53, 0.45, '1,135 cases. Schools, cafés, restaurants, sports clubs closed (15 March)', 'black'],

                     [56, 0.30, 'the minister for Medical care got sick during the debate and collapsed (18 March)', 'black']

                     ], d_rate=3.5,smooth_koef=0.0,shift_deaths_day=20,shift_confirm_day=13)
# information on Hubei

region_df = df[(df['Country/Region']=='China')&(df['Province/State']=='Hubei')]

 

plt.figure()

show_est(region_df,[[30, 0.9, '16 cases in Lombardy on 21 Feb', 'black'],

                    [len(region_df)-20, 0, 'data on last 20 days is N/A', 'orange']],d_rate=3.0)

plt.show()