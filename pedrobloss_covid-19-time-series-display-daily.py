# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

def le(path):

    df = pd.read_csv(path, sep=',')

    return df
recov = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv', 

                   sep=',')

recov.head()
recov.describe()
deaths = le('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

deaths.head()
ts_confirmed = le('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

ts_confirmed.head()
open_line = le('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

open_line.head()
line_list = le('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

line_list.head()
data = le('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
data[data['Country/Region']=='Mainland China'].describe()
data[data['Deaths']>0]['Country/Region'].unique()
data['Country/Region'].unique()
data.columns
data_br = data[data['Country/Region']=='Brazil']

data_br.head()
data_br['ObservationDate'].unique()[-1]
data_br.describe()
mortes_br = data_br['Deaths']

confirmados_br = data_br['Confirmed']

recuperados_br = data_br['Recovered']



x = data_br['ObservationDate']
recuperados_br.unique()
mortes_br.unique()
confirmados_br.unique()[-1]
plt.figure(figsize=(15,15))



plt.scatter(x, mortes_br, color='black')

plt.plot(x, mortes_br, color='red', label='Mortes')

plt.legend(loc='best')



plt.scatter(x, confirmados_br, color='black')

plt.plot(x, confirmados_br, color='blue', label='Confirmados')

plt.legend(loc='best')



plt.scatter(x, recuperados_br, color='black')

plt.plot(x, recuperados_br, color='green', label='Recuperados')

plt.legend(loc='best')



plt.xticks(x, rotation=45)

plt.show()
plt.figure(figsize=(15,15))

plt.title("Casos no Brazil até "+str())

plt.subplot(3,1,1)

plt.scatter(x, mortes_br, color='black')

plt.plot(x, mortes_br, color='red', label='Mortes')

plt.legend(loc='best')



plt.subplot(3,1,2)

plt.scatter(x, confirmados_br, color='black')

plt.plot(x, confirmados_br, color='blue', label='Confirmados')

plt.legend(loc='best')



plt.subplot(3,1,3)

plt.scatter(x, recuperados_br, color='black')

plt.plot(x, recuperados_br, color='green', label='Recuperados')

plt.legend(loc='best')



plt.xticks(x, rotation=45)

plt.tight_layout()

plt.show()


plt.figure(figsize=(15,15))



plt.title('N. de casos de COVID-19 no Brasil até '+ str(data_br['ObservationDate'].values[-1]))

plt.scatter(x, confirmados_br, color='black')

plt.plot(x, confirmados_br, color='blue', label='Confirmados')

plt.legend(loc='best')







plt.xticks(x, rotation=45)



plt.show()

def show_cases_for_country(country, rotation=80):

    

    data_country = data[data['Country/Region']==country]

     

    mortes_country = data_country['Deaths']

    confirmados_country = data_country['Confirmed']

    recuperados_country = data_country['Recovered']



    x = data_country['ObservationDate'].values

    

    last_day = str(data_country["ObservationDate"].values[-1])

    

    

    plt.figure(figsize=(15,15))

    



    plt.title(f"N. of COVID-19 cases in {country} until {last_day}", fontsize=24)

    

    plt.scatter(x, confirmados_country, color='black')

    plt.plot(x, confirmados_country, color='blue', label='Confirmed')

    plt.legend(loc='best')

    

    if mortes_country.mean() <=0:

        print(f"0 Deaths in {country}")

    else:

        plt.scatter(x, mortes_country, color='black')

        plt.plot(x, mortes_country, color='red', label='Deaths')

        plt.legend(loc='best')

    

    if recuperados_country.mean() <=0:

        print(f"0 Recovered in {country}")

    else:

        plt.scatter(x, recuperados_country, color='black')

        plt.plot(x, recuperados_country, color='green', label='Recovered')

        plt.legend(loc='best')





    plt.xticks(x, rotation=rotation)



    plt.show()

    

    

    

    

    plt.figure(figsize=(15,15))

    

    plt.subplot(3,1,1)

    plt.title(f"N. of COVID-19 cases in {country} until {last_day}", fontsize=24)

    

    plt.plot(x, confirmados_country, color='blue', label='Confirmed')

    plt.legend(loc='best')

    plt.scatter(x, confirmados_country, color='black')

    plt.xticks(x, rotation=rotation)

    

    if mortes_country.mean() <=0:

        print(f"0 Deaths in {country}")

    else:

        plt.subplot(3,1,2)



        plt.plot(x, mortes_country, color='red', label='Deaths')

        plt.legend(loc='best')

        plt.scatter(x, mortes_country, color='black')

        plt.xticks(x, rotation=rotation)

    

    if recuperados_country.mean() <=0:

        print(f"0 Recovered in {country}")

    else:

        plt.subplot(3,1,3)

        

        plt.plot(x, recuperados_country, color='green', label='Recovered')

        plt.legend(loc='best')

        plt.scatter(x, recuperados_country, color='black')

        plt.xticks(x, rotation=rotation)



    

    plt.tight_layout()

    plt.show()

data['Country/Region'].unique()
data[data['Country/Region']=='Mainland China']['Deaths'].mean()

show_cases_for_country(country='Mainland China')
show_cases_for_country(country='Italy')
data[data['Deaths']>0]['Country/Region'].unique()
show_cases_for_country(country='India')
show_cases_for_country(country='Argentina')
show_cases_for_country(country='Spain')
show_cases_for_country(country='Germany')
show_cases_for_country(country='US')
show_cases_for_country(country='Russia')
show_cases_for_country(country='Brazil')
data[data['Country/Region']=='Brazil']

