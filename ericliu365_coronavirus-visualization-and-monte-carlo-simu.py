import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random
total = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

confirm = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv')

death = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv')

recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv')
total.head()
confirm.head()
confirm = confirm.fillna(0)

death = death.fillna(0)

recovered = recovered.fillna(0)
np.any(confirm.isnull())
np.any(death.isnull())
np.any(recovered.isnull())
confirm_total = confirm.sum()[2:]

death_total = death.sum()[2:]

recovered_total = recovered.sum()[2:]
df_confirm_total = pd.DataFrame(confirm_total).reset_index()

df_confirm_total = df_confirm_total.rename(columns = {'index': 'Time', 0: 'Confirmed'})



df_death_total = pd.DataFrame(death_total).reset_index()

df_death_total = df_death_total.rename(columns = {'index': 'Time', 0: 'Death'})



df_recovered_total = pd.DataFrame(recovered_total).reset_index()

df_recovered_total = df_recovered_total.rename(columns = {'index': 'Time', 0: 'Recovered'})
df_total = pd.concat([df_confirm_total['Time'], df_confirm_total['Confirmed'], df_death_total['Death'], df_recovered_total['Recovered']], axis=1)

df_total['Time'] = pd.to_datetime(df_total['Time'])

df_total['Time'] = df_total['Time'].dt.normalize()

df_total.set_index('Time')

df_total['Death_ratio'] = df_total['Death']/df_total['Confirmed'] * 100

df_total['Recovery_ratio'] = df_total['Recovered']/df_total['Confirmed'] * 100

df_total['change_in_confirmed'] = df_total['Confirmed'].pct_change()

df_total.head()
fig = plt.figure()

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



ax1.set_title('The number of people in Confirmed, Death and Recovered', fontsize = 18)

ax1.set_xlabel('Date', fontsize = 14)

ax1.set_ylabel('The number of people', fontsize = 14)



ax2.set_xlabel('Date', fontsize = 14)

ax2.set_ylabel('Percentage (%)', fontsize = 14)

ax2.set_title('Death rate VS Recovery rate', fontsize = 18)



df_total.plot(kind = 'line', x = 'Time', y = 'Confirmed', color = 'blue', ax = ax1, figsize = (20,10))

df_total.plot(kind = 'line', x = 'Time', y = 'Death', color = 'red', ax = ax1, figsize = (20,10))

df_total.plot(kind = 'line', x = 'Time', y = 'Recovered', color = 'Green', ax = ax1, figsize = (20,10))





df_total.plot(kind = 'line', x = 'Time', y = 'Death_ratio', color = 'red', ax = ax2, figsize = (20,10))

df_total.plot(kind = 'line', x = 'Time', y = 'Recovery_ratio', color = 'Green', ax = ax2, figsize = (20,10))



fig.text(0.4,0.05, 'Data Source: Johns Hopkins University', fontsize = 10)
ini_patient = 41 

num_of_people_met_daily = 7

special_treatement_involved_day = 10  

city_closed_day = 24
death_std = np.std(df_total['Death_ratio'])/100

death_mean = np.mean(df_total['Death_ratio'])/100

recovery_std = np.std(df_total['Recovery_ratio'])/100

recovery_mean = np.mean(df_total['Recovery_ratio'])/100
def random_walk(n, t = 10 , c = 24):

    """ return the day that number of confirmed patient reduced to the half of maximum confirmed patients after breakout, n is the number of breakout day after 12/31/2020,

    t is the speical treatment involved day, c is the city closed day"""

    patient = 41

    con_patient = []

    for i in range(n):

        if i < t:

            patient =  patient * np.random.normal(6.2, 6.43) * (1 + np.random.uniform(0.014, 0.025)) * (1 - np.random.normal(death_mean, death_std))

            con_patient.append(patient)

        elif i < c:

            patient = patient * np.random.normal(6.2, 6.43) * 0.5 * (1 + np.random.uniform(0.014, 0.025)) * (1 - np.random.normal(death_mean, death_std)) * (1 - np.random.normal(recovery_mean, recovery_std))

            con_patient.append(patient)

        else: 

            patient = patient * np.random.normal(6.2, 6.43) * 0.1 * (1 + np.random.uniform(0.014, 0.025)) * (1 - np.random.normal(death_mean, death_std)) * (1 - np.random.normal(recovery_mean, recovery_std))

            con_patient.append(patient)

    max_patient = max(con_patient)

    max_patient_loc = con_patient.index(max_patient)

    for i in range(len(con_patient)):

        if i > max_patient_loc and con_patient[i] < max_patient/3:

            break

    return i + 1 - max_patient_loc
random_walk(274)
number_of_simulation = 10000

days_after_max_number = []

for i in range(number_of_simulation):

    days_after_max_number.append(random_walk(274))

days_after_max_number = pd.Series(days_after_max_number)
count_days = days_after_max_number.value_counts()

df_count_days = pd.DataFrame(count_days).reset_index()

df_count_days = df_count_days.rename(columns = {'index': 'days after the date with maximum number of confirmed patients', 0:'Frq'})
df_count_days['Probability'] = df_count_days['Frq']/number_of_simulation

df_count_days
p = df_count_days.plot.bar(x = 'days after the date with maximum number of confirmed patients', y = 'Probability', figsize = (15,8), legend = False)

p.set_title('How many days will the number of confirmed patients reduces to 50% of maximum number', fontsize = 18)

p.set_xlabel('days after the date with maximum number of confirmed patients', fontsize = 14)

p.set_ylabel('Probability', fontsize = 14)

p.text(7,0.02, 'Data Source: Johns Hopkins University', fontsize = 10)