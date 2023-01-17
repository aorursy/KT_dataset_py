import os 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# commit = 'e0ae6f6c8ab359ef6582f51453c852094255b1f9' 

commit = 'master'



url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19'

path = 'csse_covid_19_data/csse_covid_19_time_series'



for name in ['Confirmed', 'Deaths', 'Recovered']:

    print(f'{url}/{commit}/{path}/time_series_19-covid-{name}.csv')

    res = os.popen(f'curl {url}/{commit}/{path}/time_series_19-covid-{name}.csv')

    with open(f"time_series_19-covid-{name}.csv", 'w') as f:

        f.write(res.read())

        

data = {

    'confirmed': pd.read_csv("time_series_19-covid-Confirmed.csv"),

    'deaths' : pd.read_csv("time_series_19-covid-Deaths.csv"),

    'recovered' :pd.read_csv("time_series_19-covid-Recovered.csv")

}
confirm = data['confirmed']

dir(confirm)

col = 3

for i, c in enumerate(confirm[ 'Country/Region']):

    print(f"{i:>5}-'{c}'", end='' if i%col else '\n')
class Country:

    def __init__(self, data: dict, country: str):

        confirmed = data['confirmed']

        deaths = data['deaths']

        recovered = data['recovered']



        self._confirmed = confirmed.loc[confirmed['Country/Region'] == country]

        self._deaths = deaths.loc[deaths['Country/Region'] == country]

        self._recovered = recovered.loc[recovered['Country/Region'] == country]

        

        self.__data_vals = {}

        

    @property

    def vals(self):

        if not self.__data_vals:

            self.__data_vals = {

                'deaths': self._deaths[self._deaths.columns[4:]],

                'recovered':self._recovered[self._recovered.columns[4:]],

                'confirmed':self._confirmed[self._confirmed.columns[4:]]

            }

        return self.__data_vals

    

    @property

    def death(self):

        return self.vals['deaths'].T

    @property

    def recover(self):

        return self.vals['recovered'].T

    @property

    def confirm(self):

        return self.vals['confirmed'].T
def visualize(TOPIC='deaths', include_china=True, min_val=1, skip_days=0, legend=True):

    AFTER_THESE_DAYS = skip_days

    THRESHOLD = min_val

    EXCEPT_COUNTRY = 'China' if not include_china else ''

    section = data[TOPIC]

    UNTIL_THIS_DAY = len(section.loc[1]) - AFTER_THESE_DAYS-4

    days = list(section.keys()[AFTER_THESE_DAYS+4:])

    TODAY_DATE = days[-1]  # '3/10/20'

    countries = section['Country/Region']

    cx = section.loc[section[TODAY_DATE] > THRESHOLD]

    cxx = cx.loc[cx['Country/Region'] != EXCEPT_COUNTRY]

    # plt.plot(cxx.values.T[AFTER_THESE_DAYS+4:])

    fig = plt.figure(figsize=(20, 8))  



    for i, v in enumerate(cxx.values):

        args = [ '-',]

        kwargs = {'label': f'{v[1][:4]}-{str(v[0])[:4]}', 'linewidth': 3, 'alpha':0.7}

        if 'iran' in v[1].lower():

            args = ['k-']

            kwargs.update({'linewidth': 8} )

        plt.semilogy(v[AFTER_THESE_DAYS+4:], *args, **kwargs)

        plt.semilogy(v[AFTER_THESE_DAYS+4:], 'k--', linewidth=1)

#         plt.plot()



    plt.title(f'COVID-19 {TOPIC} -- regions with more than {THRESHOLD} cases')

    plt.suptitle(TOPIC)

    if legend:

        plt.legend(loc='upper left', ncol=10)

    plt.grid()

    plt.xticks(range(len(days)), days, rotation=70)

    plt.ylim(1, 1_000_000)

    plt.show()
skip = 0

visualize(TOPIC='deaths', min_val=1, include_china=False,skip_days=skip)

visualize(TOPIC='recovered',min_val=5, include_china=False,skip_days=skip)

visualize(TOPIC='confirmed',min_val=30, include_china=False,skip_days=skip)
skip = 0

visualize(TOPIC='deaths', min_val=10, skip_days=skip)

visualize(TOPIC='recovered',min_val=50, skip_days=skip)

visualize(TOPIC='confirmed',min_val=100, skip_days=skip, legend=False)
def visualize2(countries_list, skip_days=1, legend=True):



    AFTER_THESE_DAYS = skip_days

    fig = plt.figure(figsize=(20, 8))  

    THRESHOLD=1

    shapes = 'x^>os'

    shapes *= (1+ len(countries_list)//len(shapes))

    

    for ss, country in zip(shapes, countries_list):

        for topic,c in [('deaths', 'r'), ('recovered', 'g'), ('confirmed', 'b')]:







            section = data[topic]

            UNTIL_THIS_DAY = len(section.loc[1]) - AFTER_THESE_DAYS-4

            days = list(section.keys()[AFTER_THESE_DAYS+4:])

            TODAY_DATE = days[-1]  

            countries = section['Country/Region']

            cx = section.loc[section[TODAY_DATE] > 0]









            cxx = section.loc[section['Country/Region'] == country]



            for i, v in enumerate(cxx.values):

                args = [ f'{c}{ss}-',]

                kwargs = {'label': f'{topic}-{v[1][:4]}-{str(v[0])[:4]}', 'linewidth': 6, 'alpha':0.5}



                plt.semilogy(v[AFTER_THESE_DAYS+4:], *args, **kwargs)

                plt.semilogy(v[AFTER_THESE_DAYS+4:], f'k{ss}--', linewidth=1)

                plt.text(len(days)-1, v[-1], country)

        #         plt.plot()



    plt.title(f'8888888')

    plt.suptitle(country)

    if legend:

        plt.legend(loc='upper left', ncol=3)

    plt.grid()

    plt.xticks(range(len(days)), days, rotation=70)

    plt.ylim(1, 100_000)

    plt.show()
visualize2(['Iran', 'Italy', 'Korea, South','Japan'])
visualize(TOPIC='recovered',min_val=100, include_china=False,skip_days=skip)



for c in ['Iran', 'Italy', 'Korea, South','Japan', 'Korea, South']:

    visualize2([c])
iran = Country(data, 'Iran (Islamic Republic of)')

south_korea = Country(data, 'South Korea')

italy = Country(data, 'Italy')
# iran.death.plot()

# iran.recover.plot()

# iran.confirm.plot()

# plt.show()
# iran = Country(data, 'Iran (Islamic Republic of)')

# iran.recover.plot()

# plt.show()