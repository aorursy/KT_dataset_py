import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



conf = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

rec = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

dea = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"



confirmed = pd.read_csv(conf)

recovered = pd.read_csv(rec)

deaths = pd.read_csv(dea)



confirmed = np.sum(confirmed.iloc[:,4:confirmed.shape[1]])

recovered = np.sum(recovered.iloc[:,4:recovered.shape[1]])

deaths = np.sum(deaths.iloc[:,4:deaths.shape[1]])



global_mortality = (deaths/confirmed)*100



# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot for each case variable(suspected, recovered and death)

plt.plot(global_mortality

        , color = 'red'

        , label = 'Mortality Rate'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Global Mortality Rate Over the Time',size=30)

plt.ylabel('Rate',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
confirmed = pd.read_csv(conf)

recovered = pd.read_csv(rec)

deaths = pd.read_csv(dea)
# replacing missings

confirmed = confirmed.fillna('unknow')

recovered = recovered.fillna('unknow')

deaths = deaths.fillna('unknow')
# defyning the last update

last_update = '5/14/20'
#  taking total confirmed, recovered and deaths from last_update and joing



confir = confirmed[['Province/State',last_update]][confirmed['Country/Region']=='China'][last_update]

recover = recovered[last_update][recovered['Country/Region']=='China']

deat = deaths[last_update][deaths['Country/Region']=='China']



china_cases = confirmed[['Province/State',last_update]][confirmed['Country/Region']=='China']

china_cases['recovered'] = recover

china_cases['non_recovered'] = confir

china_cases['deaths'] = deat



# setting "Province/State" as index

china_cases = china_cases.set_index('Province/State')



# renaming columns

china_cases = china_cases.rename(columns = {last_update:'confirmed'

                                            ,'recovered':'recovered'

                                            ,'non_recovered':'non_recovered'

                                            ,'deaths':'deaths'})



china_cases.iloc[13,1] = 67801

china_cases.iloc[13,2] = 18
# creating the plot

china_cases.sort_values(by='confirmed',ascending=True).plot(kind='barh'

                                                            , figsize=(20,30)

                                                            , color = ['#4b8bbe','lime','orange','red']

                                                            , width=1

                                                            , rot=2)



# defyning legend and titles parameters

plt.title('Total cases by Province/State in China', size=40)

plt.ylabel('Province/State',size=30)

plt.yticks(size=20)

plt.xticks(size=20)

plt.legend(bbox_to_anchor=(0.95,0.95) # setting coordinates for the caption box

           , frameon = True

           , fontsize = 20

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# taking cases numbers

Hubei = china_cases[china_cases.index=="Hubei"]

Hubei = Hubei.iloc[0]

Hubei = Hubei.iloc[1:4]



# difyning plot size

plt.figure(figsize=(15,15))



# here i use .value_counts() to count the frequency that each category occurs of dataset

Hubei.plot(kind='pie'

           , colors=['lime','orange','red']

           , autopct='%1.1f%%' # adding percentagens

           , shadow=True

           , startangle=140)



# defyning titles and legend parameters

plt.title('Hubei Cases Distribution',size=30)

plt.legend(loc = "upper right"

           , frameon = True

           , fontsize = 15

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# creating a subset with confirmed cases in China

confirmed_china = confirmed[confirmed['Country/Region']=='China']

confirmed_china = confirmed_china.groupby(confirmed_china['Country/Region']).sum()



# taking confirmed cases growth over the time

confirmed_china = confirmed_china.iloc[0][2:confirmed_china.shape[1]]
# defyning plotsize

plt.figure(figsize=(20,10))



# creating the plot

plt.plot(confirmed_china

        , color = '#4b8bbe'

        , label = 'comfirmed'

        , marker = 'o')



# titles parameters

plt.title('Confirmed cases over the time in China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# creating a subset with recoreved cases in China

recovered_china = recovered[recovered['Country/Region']=='China']

recovered_china = recovered_china.groupby(recovered_china['Country/Region']).sum()



# taking recovered cases growth over the time

recovered_china = recovered_china.iloc[0][2:recovered_china.shape[1]]



# creating a subset with death cases in China

deaths_china = deaths[deaths['Country/Region']=='China']

deaths_china = deaths_china.groupby(deaths_china['Country/Region']).sum()



# taking death cases growth over the time

deaths_china = deaths_china.iloc[0][2:deaths_china.shape[1]]



non_recovered_china = confirmed_china-recovered_china-deaths_china
# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(recovered_china

        , color = 'lime'

        , label = 'recovered'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Recovered over the time in China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(non_recovered_china

        , color = 'orange'

        , label = 'non_recovered'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Non_Recovered over the time in China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(deaths_china

        , color = 'red'

        , label = 'death'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Deaths over the time in China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# taking mortality and recovered ratios in China

recovered_rate = (recovered_china/(confirmed_china))*100



# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(recovered_rate

        , color = 'lime'

        , label = 'recovered rate'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Recovered Rate Over the time In China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# taking mortality and recovered ratios in China

non_recovered_rate = (non_recovered_china/(confirmed_china))*100



# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(non_recovered_rate

        , color = 'Orange'

        , label = 'Non recovered rate'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Non recovered Rate Over the time In China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper right"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# taking mortality and recovered ratios in China

mortality_rate = (deaths_china/(confirmed_china))*100



# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(mortality_rate

        , color = 'red'

        , label = 'mortality rate'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Mortality Rate Over the time In China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# selecting cases by country that are not located in Mainland China

other_countries_confirmed = confirmed[confirmed.columns[4:confirmed.shape[1]]][confirmed['Country/Region']!='China']

other_countries_confirmed = other_countries_confirmed.iloc[0:other_countries_confirmed.shape[0]].sum()



other_countries_recovered = recovered[recovered.columns[4:recovered.shape[1]]][recovered['Country/Region']!='China']

other_countries_recovered = other_countries_recovered.iloc[0:other_countries_recovered.shape[0]].sum()



other_countries_deaths = deaths[deaths.columns[4:deaths.shape[1]]][deaths['Country/Region']!='China']

other_countries_deaths = other_countries_deaths.iloc[0:other_countries_deaths.shape[0]].sum()



other_countries_non_recovered = other_countries_confirmed-other_countries_recovered-other_countries_deaths
# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(other_countries_confirmed

        , color = '#4b8bbe'

        , label = 'confirmed'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Confirmed over the time outside of China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(other_countries_recovered

        , color = 'lime'

        , label = 'recovered'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Recovered over the time outside of China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(other_countries_non_recovered

        , color = 'Orange'

        , label = 'Non recovered'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Non recovered over the time outside of China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(other_countries_deaths

        , color = 'red'

        , label = 'deaths'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Deaths over the time outside of China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
recovered_rate_other_countries = (other_countries_recovered/(other_countries_confirmed))*100



# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(recovered_rate_other_countries

        , color = 'lime'

        , label = 'recovered rate'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Recovered Rate Over the time outside of China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
non_recovered_rate_other_countries = (other_countries_non_recovered/(other_countries_confirmed))*100



# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(non_recovered_rate_other_countries

        , color = 'Orange'

        , label = 'Non recovered rate'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Non recovered Rate Over the time outside of China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
death_rate_other_countries = (other_countries_deaths/(other_countries_confirmed))*100



# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot

plt.plot(death_rate_other_countries

        , color = 'red'

        , label = 'mortality rate'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Mortality Rate Over the time outside of China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# selecting cases that are not in China and join



confir = confirmed[['Country/Region','Province/State',last_update]][confirmed['Country/Region']!='China'][last_update]

recover = recovered[last_update][recovered['Country/Region']!='China']

deat = deaths[last_update][deaths['Country/Region']!='China']



other_countries = confirmed[['Country/Region','Province/State',last_update]][confirmed['Country/Region']!='China']

other_countries['recovered'] = recover

other_countries['non_recovered'] = confir-recover-deat

other_countries['deaths'] = deat



# sum the cases by country/region

other_countries = other_countries.groupby(other_countries['Country/Region']).sum()



# renaming the columns

other_countries = other_countries.rename(columns = {last_update:'confirmed'

                                                    ,'recovered':'recovered'

                                                    ,'non_recovered':'non_recovered'

                                                    ,'deaths':'deaths'})



other_countries['non_recovered'][other_countries['non_recovered']<0] = 0
# creating the plot

other_countries.sort_values(by='confirmed',ascending=True).plot(kind='barh'

                                                                , figsize=(20,50)

                                                                , color = ['#4b8bbe','lime','orange','red']

                                                                , width=1

                                                                , rot=2)



# defyning titles, labels, xticks and legend parameters

plt.title('Total cases by country', size=40)

plt.ylabel('country',size=30)

plt.yticks(size=20)

plt.xticks(size=20)

plt.legend(bbox_to_anchor=(0.95,0.95)

           , frameon = True

           , fontsize = 20

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
top_10_confirmed = confirmed[(confirmed['Country/Region']=='US') | 

                             (confirmed['Country/Region']=='Spain') |

                             (confirmed['Country/Region']=='Italy') |

                             (confirmed['Country/Region']=='France') |

                             (confirmed['Country/Region']=='Germany') |

                             (confirmed['Country/Region']=='United Kingdom') |

                             (confirmed['Country/Region']=='Turkey') |

                             (confirmed['Country/Region']=='Iran') |

                             (confirmed['Country/Region']=='Russia') |

                             (confirmed['Country/Region']=='Brazil')]



top_10_confirmed = top_10_confirmed.groupby(top_10_confirmed['Country/Region']).sum()



top_10_confirmed = top_10_confirmed.drop(['Lat','Long'], axis = 1)

top_10_confirmed = top_10_confirmed.transpose()
# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot



US = top_10_confirmed['US'][top_10_confirmed['US']>0].reset_index().drop('index',axis=1)



plt.plot(np.log(US)

        , color = 'red'

        , label = 'US'

        , marker = 'o')



Spain = top_10_confirmed['Spain'][top_10_confirmed['Spain']>0].reset_index().drop('index',axis=1)



plt.plot(np.log(Spain)

        , color = 'green'

        , label = 'Spain'

        , marker = 'o')



Italy = top_10_confirmed['Italy'][top_10_confirmed['Italy']>0].reset_index().drop('index',axis=1)



plt.plot(np.log(Italy)

        , color = 'cyan'

        , label = 'Italy'

        , marker = 'o')



France = top_10_confirmed['France'][top_10_confirmed['France']>0].reset_index().drop('index',axis=1)



plt.plot(np.log(France)

        , color = 'black'

        , label = 'France'

        , marker = 'o')



Germany = top_10_confirmed['Germany'][top_10_confirmed['Germany']>0].reset_index().drop('index',axis=1)



plt.plot(np.log(Germany)

        , color = 'blue'

        , label = 'Germany'

        , marker = 'o')



UK = top_10_confirmed['United Kingdom'][top_10_confirmed['United Kingdom']>0].reset_index().drop('index',axis=1)



plt.plot(np.log(UK)

        , color = 'darkred'

        , label = 'United Kingdom'

        , marker = 'o')



Turkey = top_10_confirmed['Turkey'][top_10_confirmed['Turkey']>0].reset_index().drop('index',axis=1)



plt.plot(np.log(Turkey)

        , color = 'darkblue'

        , label = 'Turkey'

        , marker = 'o')



Iran = top_10_confirmed['Iran'][top_10_confirmed['Iran']>0].reset_index().drop('index',axis=1)



plt.plot(np.log(Iran)

        , color = 'pink'

        , label = 'Iran'

        , marker = 'o')



Russia = top_10_confirmed['Russia'][top_10_confirmed['Russia']>0].reset_index().drop('index',axis=1)



plt.plot(np.log(Russia)

        , color = 'orange'

        , label = 'Russia'

        , marker = 'o')



Brazil = top_10_confirmed['Brazil'][top_10_confirmed['Brazil']>0].reset_index().drop('index',axis=1)



plt.plot(np.log(Brazil)

        , color = 'magenta'

        , label = 'Brazil'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Top 10 countries in log confirmed cases since first case appear',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=90,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# taking cases in Italy

Italy = other_countries[other_countries.index=="Italy"]

Italy = Italy.iloc[0]

Italy = Italy.iloc[1:4]



# taking cases in Iran

Iran = other_countries[other_countries.index=="Iran"]

Iran = Iran.iloc[0]

Iran = Iran.iloc[1:4]



# taking cases in South Korea

United_kingdom = other_countries[other_countries.index=="United Kingdom"]

United_kingdom = United_kingdom.iloc[0]

United_kingdom = United_kingdom.iloc[1:4]



# taking cases in Spain

Spain = other_countries[other_countries.index=="Spain"]

Spain = Spain.iloc[0]

Spain = Spain.iloc[1:4]



# taking cases in Germany

Germany = other_countries[other_countries.index=="Germany"]

Germany = Germany.iloc[0]

Germany = Germany.iloc[1:4]



# taking cases in US

US = other_countries[other_countries.index=="US"]

US = US.iloc[0]

US = US.iloc[1:4]



# taking cases in France

France = other_countries[other_countries.index=="France"]

France = France.iloc[0]

France = France.iloc[1:4]



# taking cases in Iran

Iran = other_countries[other_countries.index=="Iran"]

Iran = Iran.iloc[0]

Iran = Iran.iloc[1:4]
fig, axes = plt.subplots(

                     ncols=2,

                     nrows=2,

                     figsize=(15, 15))



ax1, ax2, ax3, ax4 = axes.flatten()



 # here i use .value_counts() to count the frequency that each category occurs of dataset

ax1.pie(US

        , colors=['lime','orange','red']

        , autopct='%1.1f%%' # adding percentagens

        , labels=['recovered','non_recovered','deaths']

        , shadow=True

        , startangle=140)

ax1.set_title("US Cases Distribution")



ax2.pie(Italy

           , colors=['lime','orange','red']

           , autopct='%1.1f%%' # adding percentagens

           , labels=['recovered','non_recovered','deaths']

           , shadow=True

           , startangle=140)

ax2.set_title("Italy Cases Distribution")



ax3.pie(Spain

           , colors=['lime','orange','red']

           , autopct='%1.1f%%' # adding percentagens

           , labels=['recovered','non_recovered','deaths']

           , shadow=True

           , startangle=140)

ax3.set_title("Spain Cases Distribution")



ax4.pie(Germany

           , colors=['lime','orange','red']

           , autopct='%1.1f%%' # adding percentagens

           , labels=['recovered','non_recovered','deaths']

           , shadow=True

           , startangle=140)

ax4.set_title("Germany Cases Distribution")



fig.legend(['recovered','non_recovered','deaths']

           , loc = "upper right"

           , frameon = True

           , fontsize = 15

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)



plt.show();
fig, axes = plt.subplots(

                     ncols=2,

                     nrows=2,

                     figsize=(15, 15))



ax1, ax2, ax3, ax4 = axes.flatten()



 # here i use .value_counts() to count the frequency that each category occurs of dataset

ax1.pie(France

        , colors=['lime','orange','red']

        , autopct='%1.1f%%' # adding percentagens

        , labels=['recovered','non_recovered','deaths']

        , shadow=True

        , startangle=140)

ax1.set_title("France Cases Distribution")



ax2.pie(Iran

           , colors=['lime','orange','red']

           , autopct='%1.1f%%' # adding percentagens

           , labels=['recovered','non_recovered','deaths']

           , shadow=True

           , startangle=140)

ax2.set_title("Iran Cases Distribution")



ax3.pie(United_kingdom

           , colors=['lime','orange','red']

           , autopct='%1.1f%%' # adding percentagens

           , labels=['recovered','non_recovered','deaths']

           , shadow=True

           , startangle=140)

ax3.set_title("United Kingdom Cases Distribution")



ax4.pie(Iran

           , colors=['lime','orange','red']

           , autopct='%1.1f%%' # adding percentagens

           , labels=['recovered','non_recovered','deaths']

           , shadow=True

           , startangle=140)

ax4.set_title("Iran Cases Distribution")



fig.legend(['recovered','non_recovered','deaths']

           , loc = "upper right"

           , frameon = True

           , fontsize = 15

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)



plt.show();
# creating a list with confirmed, recovered and deaths cases

list_of_tuples = list(zip(other_countries_confirmed, other_countries_recovered,other_countries_non_recovered, other_countries_deaths)) 



# creating a dataframe with this list to plot the chart

other_countries_cases_growth = pd.DataFrame(list_of_tuples, index = other_countries_confirmed.index, columns = ['confirmed', 'recovered','non_recovered','deaths'])
# creating the plot

other_countries_cases_growth.plot(kind='bar'

                                  , figsize=(20,10)

                                  , width=1

                                  , color=['#4b8bbe','lime','orange','red']

                                  , rot=2)



# defyning title, labels, ticks and legend parameters

plt.title('Growth of cases over the days in outside of China', size=30)

plt.xlabel('Updates', size=20)

plt.ylabel('Cases', size=20)

plt.xticks(rotation=90, size=15)

plt.yticks(size=15)

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
current_cases = confirmed

current_cases = current_cases[['Country/Region',last_update]]

current_cases = current_cases.groupby('Country/Region').sum().sort_values(by=last_update,ascending=False)

current_cases['recovered'] = recovered[['Country/Region',last_update]].groupby('Country/Region').sum().sort_values(by=last_update,ascending=False)

current_cases['deaths'] = deaths[['Country/Region',last_update]].groupby('Country/Region').sum().sort_values(by=last_update,ascending=False)

current_cases['non_recovered'] = current_cases[last_update]-current_cases['recovered']-current_cases['deaths']

current_cases = current_cases.rename(columns={last_update:'confirmed'

                                              ,'recovered':'recovered'

                                              ,'deaths':'deaths'

                                              ,'non_recovered':'non_recovered'})



current_cases.style.background_gradient(cmap='Reds')