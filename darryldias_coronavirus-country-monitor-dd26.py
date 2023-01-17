# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.set_option("display.max_rows", 300)
def get_diff_pc (row, str_input_previous, str_input_current):

    int_current = row[str_input_current] 

    int_previous = row[str_input_previous] 

    if int_previous == 0:

        return None

    else :

        int_diff = float(int_current - int_previous)

        return round( (int_diff / int_previous) * 100.0, 1)



def get_deaths_confirmed (row, flt_input_deaths, int_input_confirmed):

    flt_deaths = row[flt_input_deaths] 

    int_confirmed = row[int_input_confirmed] 

    if flt_deaths > 0.0:

        return round( (flt_deaths / int_confirmed) * 100.0, 1)

    else :

        return 0



def get_deaths_population (row, flt_input_deaths, flt_population_millions):

    flt_deaths = row[flt_input_deaths] 

    flt_population = row[flt_population_millions] 

    if flt_population > 0.0:

        return round(flt_deaths / flt_population, 1)

    else :

        return None

str_dataset = 'corona-virus-update-dd26' # 'corona-virus-update-dd26'   'novel-corona-virus-2019-dataset'

df_cov_deaths = pd.read_csv("../input/" + str_dataset + "/time_series_covid19_deaths_global.csv")

lst_columns = df_cov_deaths.columns.tolist() 



#lst_dates = ['3/18/20', '3/19/20', '3/20/20', '3/21/20', '3/22/20', '3/23/20', '3/24/20', '3/25/20', '3/26/20', '3/27/20', '3/28/20', '3/29/20', '3/30/20', '3/31/20', '4/01/2020']

lst_dates = lst_columns[-28:]

int_temp = len(lst_dates)



str_previous_date = lst_dates[int_temp-2]

str_current_date = lst_dates[int_temp-1]

str_previous7_date = lst_dates[int_temp-8]

print ('using dataset:', str_dataset)

print('using current date:', str_current_date)
df_cov_deaths = df_cov_deaths[ df_cov_deaths[str_current_date] > 0 ]



lst_deaths = []

for str_date in lst_dates:

    int_deaths = df_cov_deaths[str_date].sum()

    lst_deaths.append(int_deaths)



df_cov_global_deaths = pd.DataFrame({'date':lst_dates})

df_cov_global_deaths['deaths_total'] = lst_deaths



df_cov_global_active = pd.DataFrame({'date':lst_dates})

df_cov_global_active['deaths_total'] = lst_deaths



df_cov_global_deaths['deaths_total_prev1'] = df_cov_global_deaths['deaths_total'].shift(1)

df_cov_global_deaths['deaths_diff_prev1'] = df_cov_global_deaths['deaths_total'] - df_cov_global_deaths['deaths_total_prev1']

df_cov_global_deaths['deaths_diff_pc_prev1'] = round((df_cov_global_deaths['deaths_diff_prev1'] / df_cov_global_deaths['deaths_total_prev1']) * 100, 2)

df_cov_global_deaths['deaths_total_mean7'] = round(df_cov_global_deaths['deaths_total'].rolling(7).mean(),0)

df_cov_global_deaths['deaths_total_mean7_prev1'] = df_cov_global_deaths['deaths_total_mean7'].shift(1)

df_cov_global_deaths['deaths_total_mean7_diff_prev1'] = df_cov_global_deaths['deaths_total_mean7'] - df_cov_global_deaths['deaths_total_mean7_prev1']

df_cov_global_deaths['deaths_total_mean7_diff_pc_prev1'] = round((df_cov_global_deaths['deaths_total_mean7_diff_prev1'] / df_cov_global_deaths['deaths_total_mean7_prev1']) * 100, 2)

# check

df_cov_global_deaths = df_cov_global_deaths[ df_cov_global_deaths['deaths_total_mean7_diff_prev1'].notnull()]

display(df_cov_global_deaths.head(50))



plot_temp = df_cov_global_deaths.plot(x='date', y='deaths_total', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('total deaths')

pyplot.grid()



plot_temp = df_cov_global_deaths.plot.bar(x='date', y='deaths_diff_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new deaths')

pyplot.grid()



plot_temp = df_cov_global_deaths.plot.bar(x='date', y='deaths_diff_pc_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new deaths %')

pyplot.grid()



plot_temp = df_cov_global_deaths.plot(x='date', y='deaths_total_mean7', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('total deaths (7 day avg)')

pyplot.grid()



plot_temp = df_cov_global_deaths.plot.bar(x='date', y='deaths_total_mean7_diff_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new deaths (7 day avg)')

pyplot.grid()



plot_temp = df_cov_global_deaths.plot.bar(x='date', y='deaths_total_mean7_diff_pc_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new deaths % (7 day avg)')

pyplot.grid()
df_cov_deaths= pd.read_csv("../input/" + str_dataset + "/time_series_covid19_deaths_global.csv")

df_cov_deaths = df_cov_deaths[ df_cov_deaths[str_current_date] > 0 ]

df_cov_deaths = df_cov_deaths[['Province/State', 'Country/Region', str_previous7_date, str_previous_date, str_current_date]]



df_cov_country_deaths = pd.DataFrame(df_cov_deaths.groupby('Country/Region')[str_previous7_date, str_previous_date, str_current_date].sum())

df_cov_country_deaths = df_cov_country_deaths[ df_cov_country_deaths[str_current_date]>0 ]



int_current_deaths = df_cov_country_deaths[str_current_date].sum()

print('total number of deaths (' + str_current_date + '):', int_current_deaths)



int_previous_deaths = df_cov_country_deaths[str_previous_date].sum()

print('total number of deaths (' + str_previous_date + '):', int_previous_deaths)



int_diff_deaths = int_current_deaths - int_previous_deaths

flt_diff_deaths_pc = round(float(int_diff_deaths) * 100 / int_previous_deaths, 1)

print('difference: ', int_diff_deaths, '(' + str(flt_diff_deaths_pc) + '%)')



int_previous7_deaths = df_cov_country_deaths[str_previous7_date].sum()

int_diff7_deaths = int_current_deaths - int_previous7_deaths

flt_diff7_deaths_pc = round(float(int_diff7_deaths) * 100 / int_previous7_deaths, 1)



df_cov_country_deaths['deaths_diff1'] = df_cov_country_deaths[str_current_date] - df_cov_country_deaths[str_previous_date]

df_cov_country_deaths['deaths_diff1_pc'] = df_cov_country_deaths.apply(get_diff_pc, axis=1, str_input_previous=str_previous_date, str_input_current=str_current_date)



df_cov_country_deaths['deaths_diff7'] = df_cov_country_deaths[str_current_date] - df_cov_country_deaths[str_previous7_date]

df_cov_country_deaths['deaths_diff7_pc'] = df_cov_country_deaths.apply(get_diff_pc, axis=1, str_input_previous=str_previous7_date, str_input_current=str_current_date)



df_cov_country_deaths = df_cov_country_deaths.sort_values(by=['deaths_diff1_pc'], ascending=False)

df_cov_country_deaths = df_cov_country_deaths.reset_index()

df_cov_country_deaths.rename(columns={str_previous7_date:'deaths_' + str_previous7_date}, inplace=True)

df_cov_country_deaths.rename(columns={str_previous_date:'deaths_' + str_previous_date}, inplace=True)

df_cov_country_deaths.rename(columns={str_current_date:'deaths_' + str_current_date}, inplace=True)

df_cov_country_deaths.to_csv('cov_country_deaths.csv', index=False)



df_temp = df_cov_country_deaths[['Country/Region', 'deaths_' + str_previous_date, 'deaths_' + str_current_date, 'deaths_diff1', 'deaths_diff1_pc']]

df_temp.head(300)
print('total number of deaths (' + str_current_date + '):', int_current_deaths)

print('total number of deaths (' + str_previous7_date + '):', int_previous7_deaths)

print('difference: ', int_diff7_deaths, '(' + str(flt_diff7_deaths_pc) + '%)')



df_cov_country_deaths = df_cov_country_deaths.sort_values(by=['deaths_diff7_pc'], ascending=False)

df_cov_country_deaths = df_cov_country_deaths.reset_index()



df_temp = df_cov_country_deaths[['Country/Region', 'deaths_' + str_previous7_date, 'deaths_' + str_current_date, 'deaths_diff7', 'deaths_diff7_pc']]

df_temp.head(300)
df_cov_confirmed = pd.read_csv("../input/" + str_dataset + "/time_series_covid19_confirmed_global.csv")

df_cov_confirmed = df_cov_confirmed[ df_cov_confirmed[str_current_date] > 0 ]



lst_confirmed = []

for str_date in lst_dates:

    int_confirmed = df_cov_confirmed[str_date].sum()

    lst_confirmed.append(int_confirmed)



df_cov_global_confirmed = pd.DataFrame({'date':lst_dates})

df_cov_global_confirmed['confirmed_total'] = lst_confirmed



df_cov_global_active = pd.merge(df_cov_global_confirmed, df_cov_global_active, how='left', on=['date'])



df_cov_global_confirmed['confirmed_total_prev1'] = df_cov_global_confirmed['confirmed_total'].shift(1)

df_cov_global_confirmed['confirmed_diff_prev1'] = df_cov_global_confirmed['confirmed_total'] - df_cov_global_confirmed['confirmed_total_prev1']

df_cov_global_confirmed['confirmed_diff_pc_prev1'] = round((df_cov_global_confirmed['confirmed_diff_prev1'] / df_cov_global_confirmed['confirmed_total_prev1']) * 100, 2)

df_cov_global_confirmed['confirmed_total_mean7'] = round(df_cov_global_confirmed['confirmed_total'].rolling(7).mean(),0)

df_cov_global_confirmed['confirmed_total_mean7_prev1'] = df_cov_global_confirmed['confirmed_total_mean7'].shift(1)

df_cov_global_confirmed['confirmed_total_mean7_diff_prev1'] = df_cov_global_confirmed['confirmed_total_mean7'] - df_cov_global_confirmed['confirmed_total_mean7_prev1']

df_cov_global_confirmed['confirmed_total_mean7_diff_pc_prev1'] = round((df_cov_global_confirmed['confirmed_total_mean7_diff_prev1'] / df_cov_global_confirmed['confirmed_total_mean7_prev1']) * 100, 2)

# check

df_cov_global_confirmed = df_cov_global_confirmed[ df_cov_global_confirmed['confirmed_total_mean7_diff_prev1'] >= 0.0 ]

display(df_cov_global_confirmed.head(50))





plot_temp = df_cov_global_confirmed.plot(x='date', y='confirmed_total', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('total confirmed cases')

pyplot.grid()



plot_temp = df_cov_global_confirmed.plot.bar(x='date', y='confirmed_diff_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new confirmed cases')

pyplot.grid()



plot_temp = df_cov_global_confirmed.plot.bar(x='date', y='confirmed_diff_pc_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new confirmed cases %')

pyplot.grid()



plot_temp = df_cov_global_confirmed.plot(x='date', y='confirmed_total_mean7', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('total confirmed cases (7 day avg)')

pyplot.grid()



plot_temp = df_cov_global_confirmed.plot.bar(x='date', y='confirmed_total_mean7_diff_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new confirmed cases (7 day avg)')

pyplot.grid()



plot_temp = df_cov_global_confirmed.plot.bar(x='date', y='confirmed_total_mean7_diff_pc_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new confirmed cases % (7 day avg)')

pyplot.grid()

df_cov_confirmed= pd.read_csv("../input/" + str_dataset + "/time_series_covid19_confirmed_global.csv")

df_cov_confirmed = df_cov_confirmed[ df_cov_confirmed[str_current_date] > 0 ]

df_cov_confirmed = df_cov_confirmed[['Province/State', 'Country/Region', str_previous7_date, str_previous_date, str_current_date]]



df_cov_country_confirmed = pd.DataFrame(df_cov_confirmed.groupby('Country/Region')[str_previous7_date, str_previous_date, str_current_date].sum())

df_cov_country_confirmed = df_cov_country_confirmed[ df_cov_country_confirmed[str_current_date]>0 ]



int_current_confirmed = df_cov_country_confirmed[str_current_date].sum()

print('total number of confirmed (' + str_current_date + '):', int_current_confirmed)



int_previous_confirmed = df_cov_country_confirmed[str_previous_date].sum()

print('total number of confirmed (' + str_previous_date + '):', int_previous_confirmed)



int_diff_confirmed = int_current_confirmed - int_previous_confirmed

flt_diff_confirmed_pc = round(float(int_diff_confirmed) * 100 / int_previous_confirmed, 1)

print('difference: ', int_diff_confirmed, '(' + str(flt_diff_confirmed_pc) + '%)')



int_previous7_confirmed = df_cov_country_confirmed[str_previous7_date].sum()

int_diff7_confirmed = int_current_confirmed - int_previous7_confirmed

flt_diff7_confirmed_pc = round(float(int_diff7_confirmed) * 100 / int_previous7_confirmed, 1)



df_cov_country_confirmed['confirmed_diff1'] = df_cov_country_confirmed[str_current_date] - df_cov_country_confirmed[str_previous_date]

df_cov_country_confirmed['confirmed_diff1_pc'] = df_cov_country_confirmed.apply(get_diff_pc, axis=1, str_input_previous=str_previous_date, str_input_current=str_current_date)



df_cov_country_confirmed['confirmed_diff7'] = df_cov_country_confirmed[str_current_date] - df_cov_country_confirmed[str_previous7_date]

df_cov_country_confirmed['confirmed_diff7_pc'] = df_cov_country_confirmed.apply(get_diff_pc, axis=1, str_input_previous=str_previous7_date, str_input_current=str_current_date)



df_cov_country_confirmed = df_cov_country_confirmed.sort_values(by=['confirmed_diff1_pc'], ascending=False)

df_cov_country_confirmed = df_cov_country_confirmed.reset_index()

df_cov_country_confirmed.rename(columns={str_previous7_date:'confirmed_' + str_previous7_date}, inplace=True)

df_cov_country_confirmed.rename(columns={str_previous_date:'confirmed_' + str_previous_date}, inplace=True)

df_cov_country_confirmed.rename(columns={str_current_date:'confirmed_' + str_current_date}, inplace=True)

df_cov_country_confirmed.to_csv('cov_country_confirmed.csv', index=False)



df_temp = df_cov_country_confirmed[['Country/Region', 'confirmed_' + str_previous_date, 'confirmed_' + str_current_date, 'confirmed_diff1', 'confirmed_diff1_pc']]

df_temp.head(300)
print('total number of confirmed (' + str_current_date + '):', int_current_confirmed)

print('total number of confirmed (' + str_previous7_date + '):', int_previous7_confirmed)

print('difference: ', int_diff7_confirmed, '(' + str(flt_diff7_confirmed_pc) + '%)')



df_cov_country_confirmed = df_cov_country_confirmed.sort_values(by=['confirmed_diff7_pc'], ascending=False)

df_cov_country_confirmed = df_cov_country_confirmed.reset_index()



df_temp = df_cov_country_confirmed[['Country/Region', 'confirmed_' + str_previous7_date, 'confirmed_' + str_current_date, 'confirmed_diff7', 'confirmed_diff7_pc']]

df_temp.head(300)
df_cov_country = pd.merge(df_cov_country_confirmed, df_cov_country_deaths, how='left', on=['Country/Region'])

df_cov_country_active = pd.merge(df_cov_country_confirmed, df_cov_country_deaths, how='left', on=['Country/Region'])



df_cov_country['deaths_confirmed'] = df_cov_country.apply(get_deaths_confirmed, axis=1, flt_input_deaths='deaths_' + str_current_date, int_input_confirmed='confirmed_' + str_current_date)

df_cov_country = df_cov_country.sort_values(by=['deaths_confirmed'], ascending=False)

df_temp = df_cov_country[['Country/Region', 'confirmed_' + str_current_date, 'deaths_' + str_current_date, 'deaths_confirmed' ]]

df_temp.head(300)
df_country_population = pd.read_csv("../input/corona-virus-update-dd26/country_population.csv")

df_cov_country = pd.merge(df_cov_country, df_country_population, how='left', on=['Country/Region'])

df_cov_country['deaths_population'] = df_cov_country.apply(get_deaths_population, axis=1, flt_input_deaths='deaths_' + str_current_date, flt_population_millions = 'PopulationMillions')

df_cov_country = df_cov_country.sort_values(by=['deaths_population'], ascending=False)

df_cov_country = df_cov_country.reset_index()

df_temp = df_cov_country[['Country/Region', 'deaths_' + str_current_date, 'PopulationMillions', 'deaths_population' ]]

df_temp.head(300)
df_cov_recovered = pd.read_csv("../input/" + str_dataset + "/time_series_covid19_recovered_global.csv")

df_cov_recovered = df_cov_recovered[ df_cov_recovered[str_current_date] > 0 ]



lst_recovered = []

for str_date in lst_dates:

    int_recovered = df_cov_recovered[str_date].sum()

    lst_recovered.append(int_recovered)



df_cov_global_recovered = pd.DataFrame({'date':lst_dates})

df_cov_global_recovered['recovered_total'] = lst_recovered



df_cov_global_active = pd.merge(df_cov_global_active, df_cov_global_recovered, how='left', on=['date'])



df_cov_global_recovered['recovered_total_prev1'] = df_cov_global_recovered['recovered_total'].shift(1)

df_cov_global_recovered['recovered_diff_prev1'] = df_cov_global_recovered['recovered_total'] - df_cov_global_recovered['recovered_total_prev1']

df_cov_global_recovered['recovered_diff_pc_prev1'] = round((df_cov_global_recovered['recovered_diff_prev1'] / df_cov_global_recovered['recovered_total_prev1']) * 100, 2)

df_cov_global_recovered['recovered_total_mean7'] = round(df_cov_global_recovered['recovered_total'].rolling(7).mean(),0)

df_cov_global_recovered['recovered_total_mean7_prev1'] = df_cov_global_recovered['recovered_total_mean7'].shift(1)

df_cov_global_recovered['recovered_total_mean7_diff_prev1'] = df_cov_global_recovered['recovered_total_mean7'] - df_cov_global_recovered['recovered_total_mean7_prev1']

df_cov_global_recovered['recovered_total_mean7_diff_pc_prev1'] = round((df_cov_global_recovered['recovered_total_mean7_diff_prev1'] / df_cov_global_recovered['recovered_total_mean7_prev1']) * 100, 2)

# check

df_cov_global_recovered = df_cov_global_recovered[ df_cov_global_recovered['recovered_total_mean7_diff_prev1'] >= 0.0 ]

display(df_cov_global_recovered.head(50))



plot_temp = df_cov_global_recovered.plot(x='date', y='recovered_total', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('total recovered')

pyplot.grid()



plot_temp = df_cov_global_recovered.plot.bar(x='date', y='recovered_diff_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new recovered')

pyplot.grid()



plot_temp = df_cov_global_recovered.plot.bar(x='date', y='recovered_diff_pc_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new recovered %')

pyplot.grid()



plot_temp = df_cov_global_recovered.plot(x='date', y='recovered_total_mean7', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('total recovered (7 day avg)')

pyplot.grid()



plot_temp = df_cov_global_recovered.plot.bar(x='date', y='recovered_total_mean7_diff_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new recovered (7 day avg)')

pyplot.grid()



plot_temp = df_cov_global_recovered.plot.bar(x='date', y='recovered_total_mean7_diff_pc_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new recovered % (7 day avg)')

pyplot.grid()
df_cov_recovered= pd.read_csv("../input/" + str_dataset + "/time_series_covid19_recovered_global.csv")

df_cov_recovered = df_cov_recovered[ df_cov_recovered[str_current_date] > 0 ]

df_cov_recovered = df_cov_recovered[['Province/State', 'Country/Region', str_previous7_date, str_previous_date, str_current_date]]



df_cov_country_recovered = pd.DataFrame(df_cov_recovered.groupby('Country/Region')[str_previous7_date, str_previous_date, str_current_date].sum())

df_cov_country_recovered = df_cov_country_recovered[ df_cov_country_recovered[str_current_date]>0 ]



int_current_recovered = df_cov_country_recovered[str_current_date].sum()

print('total number of recovered (' + str_current_date + '):', int_current_recovered)



int_previous_recovered = df_cov_country_recovered[str_previous_date].sum()

print('total number of recovered (' + str_previous_date + '):', int_previous_recovered)



int_diff_recovered = int_current_recovered - int_previous_recovered

flt_diff_recovered_pc = round(float(int_diff_recovered) * 100 / int_previous_recovered, 1)

print('difference: ', int_diff_recovered, '(' + str(flt_diff_recovered_pc) + '%)')



int_previous7_recovered = df_cov_country_recovered[str_previous7_date].sum()

int_diff7_recovered = int_current_recovered - int_previous7_recovered

flt_diff7_recovered_pc = round(float(int_diff7_recovered) * 100 / int_previous7_recovered, 1)



df_cov_country_recovered['recovered_diff1'] = df_cov_country_recovered[str_current_date] - df_cov_country_recovered[str_previous_date]

df_cov_country_recovered['recovered_diff1_pc'] = df_cov_country_recovered.apply(get_diff_pc, axis=1, str_input_previous=str_previous_date, str_input_current=str_current_date)



df_cov_country_recovered['recovered_diff7'] = df_cov_country_recovered[str_current_date] - df_cov_country_recovered[str_previous7_date]

df_cov_country_recovered['recovered_diff7_pc'] = df_cov_country_recovered.apply(get_diff_pc, axis=1, str_input_previous=str_previous7_date, str_input_current=str_current_date)



df_cov_country_recovered = df_cov_country_recovered.sort_values(by=['recovered_diff1_pc'], ascending=False)

df_cov_country_recovered = df_cov_country_recovered.reset_index()

df_cov_country_recovered.rename(columns={str_previous7_date:'recovered_' + str_previous7_date}, inplace=True)

df_cov_country_recovered.rename(columns={str_previous_date:'recovered_' + str_previous_date}, inplace=True)

df_cov_country_recovered.rename(columns={str_current_date:'recovered_' + str_current_date}, inplace=True)

df_cov_country_recovered.to_csv('cov_country_recovered.csv', index=False)



df_temp = df_cov_country_recovered[['Country/Region', 'recovered_' + str_previous_date, 'recovered_' + str_current_date, 'recovered_diff1', 'recovered_diff1_pc']]

df_temp.head(300)
print('total number of recovered (' + str_current_date + '):', int_current_recovered)

print('total number of recovered (' + str_previous7_date + '):', int_previous7_recovered)

print('difference: ', int_diff7_recovered, '(' + str(flt_diff7_recovered_pc) + '%)')



df_cov_country_recovered = df_cov_country_recovered.sort_values(by=['recovered_diff7_pc'], ascending=False)

df_cov_country_recovered = df_cov_country_recovered.reset_index()



df_temp = df_cov_country_recovered[['Country/Region', 'recovered_' + str_previous7_date, 'recovered_' + str_current_date, 'recovered_diff7', 'recovered_diff7_pc']]

df_temp.head(300)
df_cov_global_active['active_total'] = df_cov_global_active['confirmed_total'] - df_cov_global_active['deaths_total'] - df_cov_global_active['recovered_total']

df_cov_global_active = df_cov_global_active[['date', 'active_total']]



df_cov_global_active['active_total_prev1'] = df_cov_global_active['active_total'].shift(1)

df_cov_global_active['active_diff_prev1'] = df_cov_global_active['active_total'] - df_cov_global_active['active_total_prev1']

df_cov_global_active['active_diff_pc_prev1'] = round((df_cov_global_active['active_diff_prev1'] / df_cov_global_active['active_total_prev1']) * 100, 2)

df_cov_global_active['active_total_mean7'] = round(df_cov_global_active['active_total'].rolling(7).mean(),0)

df_cov_global_active['active_total_mean7_prev1'] = df_cov_global_active['active_total_mean7'].shift(1)

df_cov_global_active['active_total_mean7_diff_prev1'] = df_cov_global_active['active_total_mean7'] - df_cov_global_active['active_total_mean7_prev1']

df_cov_global_active['active_total_mean7_diff_pc_prev1'] = round((df_cov_global_active['active_total_mean7_diff_prev1'] / df_cov_global_active['active_total_mean7_prev1']) * 100, 2)

# check

#df_cov_global_active = df_cov_global_active[ df_cov_global_active['active_total_mean7_diff_prev1'] >= 0.0 ]

df_cov_global_active = df_cov_global_active[ df_cov_global_active['active_total_mean7_diff_prev1'].notnull()]

display(df_cov_global_active.head(50))





plot_temp = df_cov_global_active.plot(x='date', y='active_total', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('total active cases')

pyplot.grid()



plot_temp = df_cov_global_active.plot.bar(x='date', y='active_diff_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new active cases')

pyplot.grid()



plot_temp = df_cov_global_active.plot.bar(x='date', y='active_diff_pc_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new active cases %')

pyplot.grid()



plot_temp = df_cov_global_active.plot(x='date', y='active_total_mean7', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('total active cases (7 day avg)')

pyplot.grid()



plot_temp = df_cov_global_active.plot.bar(x='date', y='active_total_mean7_diff_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new active cases (7 day avg)')

pyplot.grid()



plot_temp = df_cov_global_active.plot.bar(x='date', y='active_total_mean7_diff_pc_prev1', figsize=(10, 5), legend=None)

pyplot.xlabel('date')

pyplot.ylabel('new active cases % (7 day avg)')

pyplot.grid()
bln_one_country = False

if bln_one_country:

    df_cov_country_active = pd.merge(df_cov_country_active, df_cov_country_recovered, how='left', on=['Province/State'])

else:

    df_cov_country_active = pd.merge(df_cov_country_active, df_cov_country_recovered, how='left', on=['Country/Region'])



df_cov_country_active['deaths_' + str_current_date].fillna(0, inplace=True)

df_cov_country_active['recovered_' + str_current_date].fillna(0, inplace=True)

df_cov_country_active['deaths_' + str_previous_date].fillna(0, inplace=True)

df_cov_country_active['recovered_' + str_previous_date].fillna(0, inplace=True)

df_cov_country_active['deaths_' + str_previous7_date].fillna(0, inplace=True)

df_cov_country_active['recovered_' + str_previous7_date].fillna(0, inplace=True)



df_cov_country_active['active_' + str_current_date] = df_cov_country_active['confirmed_' + str_current_date] - df_cov_country_active['deaths_' + str_current_date] - df_cov_country_active['recovered_' + str_current_date]

df_cov_country_active['active_' + str_previous_date] = df_cov_country_active['confirmed_' + str_previous_date] - df_cov_country_active['deaths_' + str_previous_date] - df_cov_country_active['recovered_' + str_previous_date]

df_cov_country_active['active_' + str_previous7_date] = df_cov_country_active['confirmed_' + str_previous7_date] - df_cov_country_active['deaths_' + str_previous7_date] - df_cov_country_active['recovered_' + str_previous7_date]



if bln_one_country:

    df_cov_country_active = df_cov_country_active[['Province/State', 'Country/Region', 'active_' + str_previous7_date, 'active_' + str_previous_date, 'active_' + str_current_date]]

else:

    df_cov_country_active = df_cov_country_active[['Country/Region', 'active_' + str_previous7_date, 'active_' + str_previous_date, 'active_' + str_current_date]]



if bln_one_country:

    int_current_active = int(df_cov_country_active['active_' + str_current_date].sum() / 2)

    int_previous_active = int(df_cov_country_active['active_' + str_previous_date].sum() / 2)

    int_previous7_active = int(df_cov_country_active['active_' + str_previous7_date].sum() / 2)

else:

    int_current_active = int(df_cov_country_active['active_' + str_current_date].sum())

    int_previous_active = int(df_cov_country_active['active_' + str_previous_date].sum())

    int_previous7_active = int(df_cov_country_active['active_' + str_previous7_date].sum())



print('total number of active cases (' + str_current_date + '):', int_current_active)

print('total number of active cases (' + str_previous_date + '):', int_previous_active)



int_diff_active = int_current_active - int_previous_active

flt_diff_active_pc = round(float(int_diff_active) * 100 / int_previous_active, 1)

print('difference: ', int_diff_active, '(' + str(flt_diff_active_pc) + '%)')



int_diff7_active = int_current_active - int_previous7_active

flt_diff7_active_pc = round(float(int_diff7_active) * 100 / int_previous7_active, 1)



#if bln_one_country:

#    df_cov_country_active['Province/State'] = '**********'

#    df_cov_country_active = df_cov_country_active.reset_index()

#    df_cov_country_active = pd.concat([df_cov_active, df_cov_country_active], axis=0, sort=False)



df_cov_country_active['active_diff1'] = df_cov_country_active['active_' + str_current_date] - df_cov_country_active['active_' + str_previous_date]

df_cov_country_active['active_diff1_pc'] = df_cov_country_active.apply(get_diff_pc, axis=1, str_input_previous='active_' + str_previous_date, str_input_current='active_' + str_current_date)



df_cov_country_active['active_diff7'] = df_cov_country_active['active_' + str_current_date] - df_cov_country_active['active_' + str_previous7_date]

df_cov_country_active['active_diff7_pc'] = df_cov_country_active.apply(get_diff_pc, axis=1, str_input_previous='active_' + str_previous7_date, str_input_current='active_' + str_current_date)



df_cov_country_active = df_cov_country_active.sort_values(by=['active_diff1_pc'], ascending=False)

df_cov_country_active = df_cov_country_active.reset_index()



#df_cov_country_active.rename(columns={str_previous7_date:'active_' + str_previous7_date}, inplace=True)

#df_cov_country_active.rename(columns={str_previous_date:'active_' + str_previous_date}, inplace=True)

#df_cov_country_active.rename(columns={str_current_date:'active_' + str_current_date}, inplace=True)



df_cov_country_active.drop('index', axis=1, inplace=True)

df_cov_country_active.to_csv('cov_country_active.csv', index=False)



if bln_one_country:

    df_temp = df_cov_country_active[['Province/State', 'Country/Region', 'active_' + str_previous_date, 'active_' + str_current_date, 'active_diff1', 'active_diff1_pc']]

else:

    df_temp = df_cov_country_active[['Country/Region', 'active_' + str_previous_date, 'active_' + str_current_date, 'active_diff1', 'active_diff1_pc']]

df_temp.head(300)
print('total number of active cases (' + str_current_date + '):', int_current_active)

print('total number of active cases (' + str_previous7_date + '):', int_previous7_active)

print('difference: ', int_diff7_active, '(' + str(flt_diff7_active_pc) + '%)')



df_cov_country_active = df_cov_country_active.sort_values(by=['active_diff7_pc'], ascending=False)

#df_cov_country_active.drop('level_0', axis=1, inplace=True)

df_cov_country_active = df_cov_country_active.reset_index()



if bln_one_country:

    df_temp = df_cov_country_active[['Province/State', 'Country/Region', 'active_' + str_previous7_date, 'active_' + str_current_date, 'active_diff7', 'active_diff7_pc']]

else:

    df_temp = df_cov_country_active[['Country/Region', 'active_' + str_previous7_date, 'active_' + str_current_date, 'active_diff7', 'active_diff7_pc']]

df_temp.head(300)
if bln_one_country:

    df_cov_country = pd.merge(df_cov_country, df_cov_country_active, how='left', on=['Province/State'])

    df_cov_country.rename(columns={'Country/Region_x':'Country/Region'}, inplace=True)

else:

    df_cov_country = pd.merge(df_cov_country, df_cov_country_active, how='left', on=['Country/Region'])

    

df_cov_country['active_population'] = df_cov_country.apply(get_deaths_population, axis=1, flt_input_deaths='active_' + str_current_date, flt_population_millions = 'PopulationMillions')

#df_cov_country = df_cov_country[['Country/Region', 'active_' + str_current_date, 'PopulationMillions', 'active_population']]

df_cov_country = df_cov_country.sort_values(by=['active_population'], ascending=False)

df_cov_country = df_cov_country.reset_index()

df_cov_country.drop('index', axis=1, inplace=True)



if bln_one_country:

    df_temp = df_cov_country[['Province/State', 'Country/Region', 'active_' + str_current_date, 'PopulationMillions', 'active_population' ]]

else:

    df_temp = df_cov_country[['Country/Region', 'active_' + str_current_date, 'PopulationMillions', 'active_population' ]]

df_temp.head(300)
print('using file: time_series_covid_19_deaths.csv')

df_cov_deaths.sample(10)
print('using file: time_series_covid_19_confirmed.csv')

df_cov_confirmed.sample(10)
df_cov_country.info()

df_cov_country.sample(10)

df_country_population.head(10)