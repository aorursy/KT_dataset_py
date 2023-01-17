import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")

#data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data_sp = data[data['Country/Region'] == 'Spain']

data_it = data[data['Country/Region'] == 'Italy']

data_fr = data[data['Country/Region'] == 'France']
data_sp
first_countries = ['Italy', 'Germany', 'France', 'Spain', 'UK']

asian_countries = ['South Korea', 'Japan', 'Taiwan', 'India']

more_countries = ['Austria', 'Poland', 'Sweden', 'Portugal', 'Ireland', 'Netherlands', 'Russia']

well_countries = []

countries = first_countries + more_countries + well_countries



#data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")

data = data.rename(columns={'Confirmed': 'NumberOfConfirmedCases'})

data['DayOfYear'] = pd.to_datetime(data['Date']).dt.dayofyear

data = data.filter(['DayOfYear', 'Country/Region', 'Province/State', 'NumberOfConfirmedCases', 'Deaths', 'Date', 'Recovered'])

data_ch = data[data['Country/Region'] == 'Mainland China']

data = data[data.DayOfYear >= 40]

data_us = data[data['Country/Region'] == 'US']



# only country data:

data = data[data['Province/State'].isnull() | (data['Province/State'] == data['Country/Region'])]

data['Symptomatic'] = data['NumberOfConfirmedCases'] - data['Deaths'] - data['Recovered']

data['EverSymptomatic'] = data['NumberOfConfirmedCases']








#data = data[pd.isnull(data['Province/State'])]



with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(1, figsize=(16, 20))

    ax.set_yscale('log')

    #ax.set_ylim(1, 1e5)

    ax.set_xlim(50, 85)

    df=data[data['Country/Region'] == 'Spain'].copy()

    df['NewSymptomatic'] = df['EverSymptomatic'].diff()

    df.plot(x='DayOfYear', y=['EverSymptomatic', 'Recovered', 'Symptomatic', 'Deaths', "NewSymptomatic"],

                  ax=ax, marker='o')

   
with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(2, figsize=(16, 20))

    axs[0].set_yscale('log')

    axs[0].set_ylim(1, 1e5)

    #axs[0].set_xlim(50, 78)

    axs[1].set_yscale('log')

    axs[1].set_ylim(1, 1e5)

    #axs[1].set_xlim(50, 78)

    sns.lineplot(data=data[data['Country/Region'].isin(first_countries + more_countries)],

                 x ='DayOfYear', y='NumberOfConfirmedCases', hue='Country/Region', ax=axs[0], marker='o')

    sns.lineplot(data=data[data['Country/Region'].isin(asian_countries)],

                 x ='DayOfYear', y='NumberOfConfirmedCases', hue='Country/Region', ax=axs[1], marker='o')
with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(figsize=(16, 15))

    #ax.set_yscale('log')

    grid = sns.scatterplot(data=data[data['Country/Region'].isin(countries)],

                                     x ='DayOfYear', y='Deaths', hue='Country/Region', ax=ax)
data_us = data_us[data_us.NumberOfConfirmedCases > 10]



with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(figsize=(16, 15))

    ax.set_yscale('log')

    grid = sns.lineplot(data=data_us, x ='DayOfYear', y='NumberOfConfirmedCases', hue='Province/State', ax=ax, marker='o')

from math import log, exp

from functools import reduce



with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(2, figsize=(20, 20))

    axs[0].set_yscale('log')

    axs[1].set(ylim=(-0.0, 160.0))

    



    #sns.scatterplot(data=data_es, x ='DayOfYear', y='Expected', ax=ax)

    #sns.scatterplot(data=data_es, x ='DayOfYear', y='NOCC', ax=ax)



    pandas = []

    for country in ['South Korea', 'Italy', 'Spain']: #first_countries + asian_countries:

        velocity_expected = []

        velocity = []

        ys = []

        country_data = data[data['Country/Region'] == country].copy()

        for nocc in country_data['NumberOfConfirmedCases'].values:

            ys.insert(0, log(nocc))

            if len(ys) > 1:

                n = 0;

                Sx = 0;

                Sy = 0;

                Sx2 = 0;

                Sy2 = 0;

                Sxy = 0;            

                for x, y in enumerate(ys):

                    f = 1 if x < 2 else 0

                    #f = 0.8 ** x if x < 7 else 0

                    #f = 0.9 ** x if x < 8 else 0



                    n += f

                    Sx += f * x

                    Sy += f * y

                    Sx2 += f * x * x

                    Sy2 += f * y * y

                    Sxy += f * x * y

                den = n*Sx2 - Sx*Sx

                #print("y: %s, den: %f" % (ys, den))

                a = (Sy*Sx2 - Sx*Sxy) / den

                b = (n*Sxy - Sx*Sy) / den

            

                expected = exp(a)

                expected_yesterday = exp(a + b)

                vel = expected/expected_yesterday

                #print("nocc: %f, y: %f, expected: %f, yesterday: %f, a: %f, b: %f, vel: %f" %

                #      (nocc, y, expected, expected_yesterday, a, b, vel))

                velocity_expected.append(100 * (expected/expected_yesterday - 1))

                velocity.append(100*(exp(ys[0]-ys[1]) - 1))

            else:

                velocity_expected.append(0.0)

                velocity.append(0.0)

    

        #print("len data: %s, len vel: %s" % (len(country_data), len(velocity)))

        country_data['Velocity'] = velocity_expected

        pandas.append(country_data)



    result = reduce(lambda a, b: a.append(b), pandas)

    result = result[result.DayOfYear > 50]

    sns.lineplot(data=result, x = 'DayOfYear', y='NumberOfConfirmedCases', ax=axs[0], hue='Country/Region', marker='o')

    g = sns.lineplot(data=result, x ='DayOfYear', y='Velocity', ax=axs[1],  hue='Country/Region', marker="o")

    g.set(ylabel = 'Velocity (% Daily confirmed cases increase)')

#result2 = result

result[result['DayOfYear']> 72]
#with sns.axes_style("whitegrid"):

#    fig, ax = plt.subplots(figsize=(20, 16))

#    ax.set(ylim=(-0.0, 60.0))

#    sns.lineplot(data=result[result['Country/Region'] =='Spain'], x ='DayOfYear', y='Velocity', ax=ax, marker="o")

#    sns.lineplot(data=result2[result2['Country/Region'] =='Spain'], x ='DayOfYear', y='Velocity', ax=ax, marker="o")

    
import pandas as pd

ccaa_source_data = pd.read_csv("/kaggle/input/covid19spaindata7/COVID19-Spain/ccaa_covid19_casos.csv")

ccaa_source_data=ccaa_source_data[ccaa_source_data['20/03/2020'] >= 100]

ccaa_source_data
from datetime import datetime

from functools import reduce

cas = []

cases = []

dates = []

for i, row in ccaa_source_data.iterrows():

    ca = row.CCAA

    #if ca == 'Total':

    #    ca = 'España'

    for i, val in enumerate(row):

        if i <= 2:

            continue

        date = ccaa_source_data.columns[i]

        cas.append(ca)

        cases.append(val)

        dates.append(datetime.strptime(date, "%d/%m/%Y"))



ccaa_data = pd.DataFrame({'CA': cas, 'ObservationDate': dates, 'NumberOfConfirmedCases': cases})

ccaa_data['DayOfYear'] = pd.to_datetime(ccaa_data['ObservationDate']).dt.dayofyear

ca_names = ccaa_source_data.CCAA.values
with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(figsize=(16, 15))

    ax.set_yscale('log')

    ax.set_ylim(1, 1e5)

    sns.lineplot(data=ccaa_data,

                 x ='DayOfYear', y='NumberOfConfirmedCases', hue='CA', ax=ax, marker='o')
from math import log, exp

from functools import reduce



soften_length = 4



pandas = []

with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(1, 1, figsize=(20, 16))

    #for ax in axs:

    axs.set(ylim=(1, 70.0))

    for ca in ca_names:

        soften_velocity = []

        velocity = []

        ys = []

        ca_data = ccaa_data[ccaa_data['CA'] == ca].copy()

        #print(ca, ca_data)

        for nocc in ca_data['NumberOfConfirmedCases'].values:

            y = log(nocc) if nocc >= 1 else 0

            if ys:

                ys.insert(0, y)

            else:

                ys = [y, y, y]

            n = 0;

            Sx = 0;

            Sy = 0;

            Sx2 = 0;

            Sy2 = 0;

            Sxy = 0;            

            for x, y in enumerate(ys):

                #f = 0.6 if x < 10 else 0

                #f = 0.7 ** x if x < 5 else 0

                f = 1 if x < soften_length else 0

                n += f

                Sx += f * x

                Sy += f * y

                Sx2 += f * x * x

                Sy2 += f * y * y

                Sxy += f * x * y

            den = n*Sx2 - Sx*Sx

            a = (Sy*Sx2 - Sx*Sxy) / den

            b = (n*Sxy - Sx*Sy) / den

            

            expected = exp(a)

            expected_yesterday = exp(a + b)

            vel = expected/expected_yesterday

            #print("nocc: %f, y: %f, expected: %f, yesterday: %f, a: %f, b: %f, vel: %f" %

            #      (nocc, y, expected, expected_yesterday, a, b, vel))

            soften_velocity.append(100 * (expected/expected_yesterday - 1))

            velocity.append(100*(exp(ys[0]-ys[1]) - 1))

    

        #print("len data: %s, len vel: %s" % (len(country_data), len(velocity)))

        #ca_data['SoftenVelocity'] = soften_velocity

        #ca_data['Velocity'] = velocity

        ca_data['Velocity'] = soften_velocity

        if ca == 'Total' or ca == 'Madrid':

            pandas.append(ca_data)

    

    vel_data = reduce(lambda a, b: a.append(b), pandas)

    #sns.lineplot(data=vel_data, x ='DayOfYear', y='Velocity', ax=axs[1],  hue='CA')

    sns.lineplot(data=vel_data, x ='DayOfYear', y='Velocity', ax=axs,  hue='CA', marker='o')




