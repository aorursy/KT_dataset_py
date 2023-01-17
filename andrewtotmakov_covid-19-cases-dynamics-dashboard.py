# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



%matplotlib inline
df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

df.Date = pd.to_datetime(df.Date)

#df.head(5)
# df[df.Date == df.Date.max()].groupby('Country/Region').Confirmed.sum().sort_values(ascending=False).head(30)
# Select top 5 countries with confirmed cases excluding China & Russia.

top5_countries = df[(df['Country/Region'] != 'China') & (df['Country/Region'] != 'Russia') & (df.Date == df.Date.max())].groupby('Country/Region').Confirmed.sum().sort_values(ascending=False).head(5).index
from collections import Iterable





def plot_country_stat(country, ax1, ax2, ax3):

    """

    Utility function. Plots three charts for the particluar country:

    

       * Number of confirmed, deaths and recovered cases over time with stacking to see proportions change

       * Cases change over time (first derivative) to vizualize trends change over time

       * Tension level for a healthcare system of the country: 

       

             Tesion(t) = Confirmed(t) - Deaths(t) - Recovered(t)

             

         Shows number of currenltry ill people on a date, helps to vizualize spike

    """

    cn = df[(df['Country/Region'] == country) & (df.Confirmed > 0)]

    cn_stat = cn.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].agg(np.sum)



    cn_stat['Rest'] = cn_stat.Confirmed - cn_stat.Deaths - cn_stat.Recovered

    

    ax1.stackplot(cn_stat.index, [cn_stat.Deaths, cn_stat.Recovered, cn_stat.Rest], 

                    labels=['Deaths', 'Recovered', 'Confirmed'])

    ax1.set_title('{} :: Cases'.format(country))

    ax1.legend(loc='upper left')

    plt.setp( ax1.xaxis.get_majorticklabels(), rotation=45 )

    

    ax2.plot(cn_stat[['Deaths', 'Recovered', 'Confirmed']].diff())

    ax2.legend(loc='upper left', labels=['Deaths', 'Recovered', 'Confirmed'])

    ax2.set_title('{} :: Velocity (derivative)'.format(country))

    plt.setp( ax2.xaxis.get_majorticklabels(), rotation=45 )

    

    ax3.stackplot(cn_stat.index, cn_stat.Rest)

    ax3.set_title('{} :: Tension Level'.format(country))

    plt.xticks(rotation=45)

    plt.setp( ax3.xaxis.get_majorticklabels(), rotation=45 )

    



def plot_countries(countries):

    """

    Plots charts for countries. 

    

    :param countries: can be a string with a name of country or a list of country names.

    :type countries: a string or a list

    """

    plot_title = 'COVID-19 Countries Status by Date: {}'.format(df.Date.max().date())

    plot_footer = '2020 Visualization by Nabla Analytica'

    

    if not isinstance(countries, list):

        fig, ax = plt.subplots(1, 3, figsize=(16,4))

        fig.suptitle(plot_title, y=1.08)

        plot_country_stat(countries, ax[0], ax[1], ax[2])

    else:

        countries_count = len(countries)

        fig, ax = plt.subplots(countries_count, 3, figsize=(16,countries_count*5))

        

        for i in range(0, countries_count):

            plot_country_stat(countries[i], ax[i, 0], ax[i, 1], ax[i, 2])

            

        fig.subplots_adjust(hspace=0.6)

        fig.suptitle(plot_title, fontsize=20, y=0.93)

        

    

    plt.figtext(0.92, 0.3, plot_footer, rotation='vertical')

# To supress FutureWarning

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



# Plotting charts for China (as they passed through a spike), Russia and top 5 coutries on confirmed cases

plot_countries(['China', 'Russia']) # + list(top5_countries))
plot_countries(['Austria', 'United Kingdom'])
plot_countries(list(top5_countries))