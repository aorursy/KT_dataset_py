import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        
# loading dataset

cov = pd.read_csv('../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200206.csv'

                  , header=0

                  , names=['state','country','last_update','confirmed','suspected','recovered','death'])

cov.head()
# dealing with dates, converting dates with hours and minutes in just dates with year, month and day

cov['last_update'] = pd.to_datetime(cov['last_update']).dt.date
cov.info() # seeing dataset structure
cov.isna().sum()
# replacing state missing values by "unknow"

cov['state'] = cov['state'].fillna('unknow')



# replacing numerical variables missing values by 0

cov = cov.fillna(0)
# taking cov columns where the country are China

china = cov[['state','confirmed','suspected','recovered','death']][cov['country']=='Mainland China']



# taking the max value by state

china = china[['confirmed','suspected','recovered','death']].groupby(cov['state']).max()
# creating the plot

china.sort_values(by='confirmed',ascending=True).plot(kind='barh',figsize=(20,30), width=1,rot=2)



# defyning legend and titles parameters

plt.title('Total cases by state in China',size=40)

plt.ylabel('state',size=30)

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

Hubei = china[china.index=="Hubei"]

Hubei = Hubei.iloc[0]



# difyning plot size

plt.figure(figsize=(15,15))



# here i use .value_counts() to count the frequency that each category occurs of dataset

Hubei.plot(kind='pie'

           , colors=['#4b8bbe','orange','lime','red']

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
# taking cases and dates in china

china_cases_grow = cov[['last_update','confirmed','suspected','recovered','death']][cov['country']=='Mainland China']



# creating a new subset with cases over the days

china_confirmed_grow = china_cases_grow[['confirmed']].groupby(cov['last_update']).max()

china_suspected_grow = china_cases_grow[['suspected']].groupby(cov['last_update']).max()

china_recovered_grow = china_cases_grow[['recovered']].groupby(cov['last_update']).max()

china_death_grow = china_cases_grow[['death']].groupby(cov['last_update']).max()
# defyning plotsize

plt.figure(figsize=(20,10))



# creating the plot

sns.lineplot(x = china_confirmed_grow.index

        , y = 'confirmed'

        , color = '#4b8bbe'

        , label = 'comfirmed'

        , marker = 'o'

        , data = china_confirmed_grow)



# titles parameters

plt.title('Growth of confirmed cases in China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=45,size=15)

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
# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot for each case variable(suspected, recovered and death)

sns.lineplot(x = china_suspected_grow.index

        , y = 'suspected'

        , color = 'orange'

        , label = 'suspected'

        , marker = 'o'

        , data = china_suspected_grow)



sns.lineplot(x = china_recovered_grow.index

        , y = 'recovered'

        , color = 'green'

        , label = 'recovered'

        , marker = 'o'

        , data = china_recovered_grow)



sns.lineplot(x = china_death_grow.index

        , y = 'death'

        , color = 'red'

        , label = 'death'

        , marker = 'o'

        , data = china_death_grow)



# defyning titles, labels and ticks parameters

plt.title('Growth of others case statistics in China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=45,size=15)

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
# taking all cases that are not in China

other_countries = cov[['country','confirmed','suspected','recovered','death']][cov['country']!='Mainland China']



# taking cases by country

other_countries = other_countries[['confirmed','suspected','recovered','death']].groupby(other_countries['country']).max()
# creating the plot

other_countries.sort_values(by='suspected',ascending=True).plot(kind='barh',figsize=(20,30), width=1,rot=2)



# defyning titles, labels, xticks and legend parameters

plt.title('Total cases in other countries',size=40)

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
# taking cases in Hong-Kong

Hong_Kong = other_countries[other_countries.index=="Hong Kong"]

Hong_Kong = Hong_Kong.iloc[0]



# difyning plot size

plt.figure(figsize=(15,15))



 # here i use .value_counts() to count the frequency that each category occurs of dataset

Hong_Kong.plot(kind='pie'

           , colors=['#4b8bbe','orange','lime','red']

           , autopct='%1.1f%%' # adding percentagens

           , shadow=True

           , startangle=140)



# defyning title and legend parameters

plt.title('Hong-Kong Cases Distribution',size=30)

plt.legend(loc = "upper right"

           , frameon = True

           , fontsize = 15

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
# taking cases from countries that are not China

other_countries_cases = cov[['country','last_update','confirmed','suspected','recovered','death']][cov['country']!='Mainland China']



# Taking total of confirmed cases countries

other_countries_confirmed = other_countries_cases[['last_update','confirmed']].groupby(cov['country']).max()

other_countries_cases_growth = other_countries_confirmed[['confirmed']].groupby(other_countries_confirmed['last_update']).sum()

other_countries_cases_growth = other_countries_cases_growth[['confirmed']].cumsum()



# Taking total of suspected cases countries

other_countries_suspected = other_countries_cases[['last_update','suspected']].groupby(cov['country']).max()

other_countries_suspected_growth = other_countries_suspected[['suspected']].groupby(other_countries_suspected['last_update']).sum()

other_countries_suspected_growth = other_countries_suspected_growth[['suspected']].cumsum()



# Taking total of recovered cases countries

other_countries_recovered = other_countries_cases[['last_update','recovered']].groupby(cov['country']).max()

other_countries_recovered_growth = other_countries_recovered[['recovered']].groupby(other_countries_recovered['last_update']).sum()

other_countries_recovered_growth = other_countries_recovered_growth[['recovered']].cumsum()



# Taking total of death cases countries

other_countries_death = other_countries_cases[['last_update','death']].groupby(cov['country']).max()

other_countries_death_growth = other_countries_death[['death']].groupby(other_countries_death['last_update']).sum()

other_countries_death_growth = other_countries_death_growth[['death']].cumsum()



# Joing all case types in one dataset adding new columns

other_countries_cases_growth['suspected'] = other_countries_suspected_growth['suspected']

other_countries_cases_growth['recovered'] = other_countries_recovered_growth['recovered']

other_countries_cases_growth['death'] = other_countries_death_growth['death']
# creating the plot

other_countries_cases_growth.plot(kind='bar',figsize=(20,10), width=1,rot=2)



# defyning title, labels, ticks and legend parameters

plt.title('Growth of cases over the days in other countries',size=30)

plt.xlabel('Updates', size=20)

plt.ylabel('Cases', size=20)

plt.xticks(rotation=45, size=15)

plt.yticks(size=15)

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);