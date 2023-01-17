import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import csv

import os
os.chdir('/kaggle/input/novel-corona-virus-2019-dataset/')

os.listdir()
covid_data = pd.read_csv("covid_19_data.csv")

covid_data.head()
states = covid_data['Province/State'].unique()

countries = covid_data['Country/Region'].unique()
covid_data['Country/Region'].value_counts()
# to get the last updated date of the dataset



covid_data['Last Update'].unique()
confirm_dict = {}

deaths_dict = {}

recover_dict = {}

for country in countries:

    country_data = covid_data[covid_data['Country/Region'] == country]

    #cummulative, so we can simply take the latest date for final result

    max_date = country_data['ObservationDate'].max()

    sub = country_data[country_data['ObservationDate'] == max_date]

    confirm = sub['Confirmed'].sum()

    death = sub['Deaths'].sum()

    recover = sub['Recovered'].sum()

    

    confirm_dict[country] = confirm

    deaths_dict[country] = death

    recover_dict[country] = recover

confirm_dict_sorted = sorted(confirm_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)

deaths_dict_sorted = sorted(deaths_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)

recover_dict_sorted = sorted(recover_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
top10_confirm = confirm_dict_sorted[:10]

top10_deaths = deaths_dict_sorted[:10]

top10_recover = recover_dict_sorted[:10]

top10_confirm = dict(top10_confirm)

top10_deaths = dict(top10_deaths)

top10_recover = dict(top10_recover)
plt.figure(figsize = (7,6))

bars = plt.bar(top10_confirm.keys(), top10_confirm.values())

plt.xlabel('Country')

plt.ylabel('Count')

plt.title('Highest Confirmed Cases in 10 countries')

plt.xticks(list(top10_confirm.keys()), rotation = 90)

for bar in bars:

    yval = bar.get_height()

    plt.text(bar.get_x(), yval + .005, yval, rotation = 5)

plt.show()
plt.figure(figsize = (7,6))

bars = plt.bar(top10_deaths.keys(), top10_deaths.values())

plt.xlabel('Country')

plt.ylabel('Count')

plt.title('Highest Death Cases in 10 countries')

plt.xticks(list(top10_deaths.keys()), rotation = 90)

for bar in bars:

    yval = bar.get_height()

    plt.text(bar.get_x(), yval + .005, yval, rotation = 5)

plt.show()
plt.figure(figsize = (7,6))

bars = plt.bar(top10_recover.keys(), top10_recover.values())

plt.xlabel('Country')

plt.ylabel('Count')

plt.title('Highest Recovered Cases in 10 countries')

plt.xticks(list(top10_recover.keys()), rotation = 90)

for bar in bars:

    yval = bar.get_height()

    plt.text(bar.get_x(), yval + .005, yval, rotation = 5)

plt.show()
china_data = covid_data[covid_data['Country/Region'] == 'Mainland China']

date = []

c = []

d = []

r = []

for dat in china_data['ObservationDate'].unique():

    sub = china_data[china_data['ObservationDate'] == dat]

    confirm = sub['Confirmed'].sum()

    death = sub['Deaths'].sum()

    recover = sub['Recovered'].sum()

    date.append(dat)

    c.append(confirm)

    d.append(death)

    r.append(recover)

    

date = pd.Series(date)

c  =pd.Series(c)

d = pd.Series(d)

r = pd.Series(r)



t = [date.min(), date[len(date)//2], date.max()]

plt.figure(figsize=(8,8))

plt.plot(date, c, color = 'yellow')

plt.plot(date, d, color = 'red')

plt.plot(date, r, color = 'green')

plt.xticks(t, t)

plt.xlabel('Date')

plt.ylabel('Cummulative Count cases')

plt.title('Trend Curve of Confirmed Cases in China')

plt.legend(['Confirmed', 'Death', 'Recovered'])

plt.show()
italy_data = covid_data[covid_data['Country/Region'] == 'Italy']

date = []

c = []

d = []

r = []

for dat in italy_data['ObservationDate'].unique():

    sub = italy_data[italy_data['ObservationDate'] == dat]

    confirm = sub['Confirmed'].sum()

    death = sub['Deaths'].sum()

    recover = sub['Recovered'].sum()

    date.append(dat)

    c.append(confirm)

    d.append(death)

    r.append(recover)



date = pd.Series(date)

c  =pd.Series(c)

d = pd.Series(d)

r = pd.Series(r)



t = [date.min(), date[len(date)//2], date.max()]

plt.figure(figsize=(8,8))

plt.plot(date, c, color = 'yellow')

plt.plot(date, d, color = 'red')

plt.plot(date, r, color = 'green')

plt.xticks(t, t)

plt.xlabel('Date')

plt.ylabel('Cummulative Count cases')

plt.title('Trend Curve of Confirmed Cases in Italy')

plt.legend(['Confirmed', 'Death', 'Recovered'])

plt.show()
us_data = covid_data[covid_data['Country/Region'] == 'US']

date = []

c = []

d = []

r = []

for dat in us_data['ObservationDate'].unique():

    sub = us_data[us_data['ObservationDate'] == dat]

    confirm = sub['Confirmed'].sum()

    death = sub['Deaths'].sum()

    recover = sub['Recovered'].sum()

    date.append(dat)

    c.append(confirm)

    d.append(death)

    r.append(recover)



date = pd.Series(date)

c  =pd.Series(c)

d = pd.Series(d)

r = pd.Series(r)



t = [date.min(), date[len(date)//2], date.max()]

plt.figure(figsize=(8,8))

plt.plot(date, c, color = 'yellow')

plt.plot(date, d, color = 'red')

plt.plot(date, r, color = 'green')

plt.xticks(t, t)

plt.xlabel('Date')

plt.ylabel('Cummulative Count cases')

plt.title('Trend Curve of Confirmed Cases in US')

plt.legend(['Confirmed', 'Death', 'Recovered'])

plt.show()
spain_data = covid_data[covid_data['Country/Region'] == 'Spain']

date = []

c = []

d = []

r = []

for dat in spain_data['ObservationDate'].unique():

    sub = spain_data[spain_data['ObservationDate'] == dat]

    confirm = sub['Confirmed'].sum()

    death = sub['Deaths'].sum()

    recover = sub['Recovered'].sum()

    date.append(dat)

    c.append(confirm)

    d.append(death)

    r.append(recover)



date = pd.Series(date)

c  =pd.Series(c)

d = pd.Series(d)

r = pd.Series(r)



t = [date.min(), date[len(date)//2], date.max()]

plt.figure(figsize=(8,8))

plt.plot(date, c, color = 'yellow')

plt.plot(date, d, color = 'red')

plt.plot(date, r, color = 'green')

plt.xticks(t, t)

plt.xlabel('Date')

plt.ylabel('Cummulative Count cases')

plt.title('Trend Curve of Confirmed Cases in Spain')

plt.legend(['Confirmed', 'Death', 'Recovered'])

plt.show()
germany_data = covid_data[covid_data['Country/Region'] == 'Germany']

date = []

c = []

d = []

r = []

for dat in germany_data['ObservationDate'].unique():

    sub = germany_data[germany_data['ObservationDate'] == dat]

    confirm = sub['Confirmed'].sum()

    death = sub['Deaths'].sum()

    recover = sub['Recovered'].sum()

    date.append(dat)

    c.append(confirm)

    d.append(death)

    r.append(recover)



date = pd.Series(date)

c  =pd.Series(c)

d = pd.Series(d)

r = pd.Series(r)



t = [date.min(), date[len(date)//2], date.max()]

plt.figure(figsize=(8,8))

plt.plot(date, c, color = 'yellow')

plt.plot(date, d, color = 'red')

plt.plot(date, r, color = 'green')

plt.xticks(t, t)

plt.xlabel('Date')

plt.ylabel('Cummulative Count cases')

plt.title('Trend Curve of Confirmed Cases in Germany')

plt.legend(['Confirmed', 'Death', 'Recovered'])

plt.show()
total_confirmed = sum(list(confirm_dict.values()))

total_deaths = sum(list(deaths_dict.values()))

total_recovered = sum(list(recover_dict.values()))



total_still_affected = total_confirmed -(total_deaths+total_recovered)

print("World Population affectedas of 22nd March 2020: ", total_confirmed)
groups = ['Affected and Uncured', 'Deaths', 'Recovered']

sizes = [total_still_affected, total_deaths, total_recovered]

colours = ['Yellow', 'Red', 'Green']

explode = (0, 0.2, 0)

col_labels = ['Count']

row_labels = ['Affected and Uncured', 'Deaths', 'Recovered']

table_values = [[total_still_affected],[total_deaths], [total_recovered]]





fig, axs = plt.subplots(1,2, figsize = (9,9))

axs[0].axis('tight')

axs[0].axis('off')

the_table = axs[0].table(cellText=table_values,colWidths = [0.5], colLabels=col_labels, rowLabels = row_labels, loc='center')

the_table.set_fontsize(14)

the_table.scale(1.5, 1.5)

axs[1].pie(sizes, labels = groups, explode = explode, colors=colours, shadow=True, autopct='%1.1f%%')

plt.title('Distribution at world level')

plt.show()
china_c_number = confirm_dict['Mainland China']

italy_c_number = confirm_dict['Italy']

others_c = 0

for key in confirm_dict:

    if key != 'Mainland China' and key != 'Italy':

        others_c+=confirm_dict[key]

        

groups = ['China', 'Italy', 'Others']

sizes = [china_c_number, italy_c_number, others_c]

colours = ['Red', 'Green', 'Grey']

explode = (0.1, 0, 0)

col_labels = ['Count']

row_labels = ['China', 'Italy', 'Others']

table_values = [[china_c_number], [italy_c_number], [others_c]]





fig, axs = plt.subplots(1,2, figsize = (8,8))

axs[0].axis('tight')

axs[0].axis('off')

the_table = axs[0].table(cellText=table_values,colWidths = [0.5], colLabels=col_labels, rowLabels = row_labels, loc='center')

the_table.set_fontsize(14)

the_table.scale(1.5, 1.5)

axs[1].pie(sizes, labels = groups, explode = explode, colors=colours, shadow=True, autopct='%1.1f%%')

plt.title('Global Proportions of 2 severely striken countries')

plt.show()
china_r_number = recover_dict['Mainland China']

italy_r_number = recover_dict['Italy']

others_r = 0

for key in recover_dict:

    if key != 'Mainland China' and key != 'Italy':

        others_r+=recover_dict[key]

        

groups = ['China', 'Italy', 'Others']

sizes = [china_r_number, italy_r_number, others_r]

colours = ['Red', 'Green', 'Grey']

explode = (0.1, 0, 0)

col_labels = ['Count']

row_labels = ['China', 'Italy', 'Others']

table_values = [[china_r_number], [italy_r_number], [others_r]]





fig, axs = plt.subplots(1,2, figsize = (8,8))

axs[0].axis('tight')

axs[0].axis('off')

the_table = axs[0].table(cellText=table_values,colWidths = [0.5], colLabels=col_labels, rowLabels = row_labels, loc='center')

the_table.set_fontsize(14)

the_table.scale(1.5, 1.5)

axs[1].pie(sizes, labels = groups, explode = explode, colors=colours, shadow=True, autopct='%1.1f%%')

plt.title('Global Proportions of 2 severely striken countries')

plt.show()
italian_death_perc = (deaths_dict['Italy']/total_deaths)*100

print('Death Percentage in Italy: ', italian_death_perc)



china_death_perc = (deaths_dict['Mainland China']/total_deaths)*100

print('Death Percentage in China: ', china_death_perc)



print(total_deaths)
china_d_number = deaths_dict['Mainland China']

italy_d_number = deaths_dict['Italy']

others_d = 0

for key in deaths_dict:

    if key != 'Mainland China' and key != 'Italy':

        others_d+=deaths_dict[key]

        

groups = ['China', 'Italy', 'Others']

sizes = [china_d_number, italy_d_number, others_d]

colours = ['Red', 'Green', 'Grey']

explode = (0.1, 0, 0)

col_labels = ['Count']

row_labels = ['China', 'Italy', 'Others']

table_values = [[china_d_number], [italy_d_number], [others_d]]





fig, axs = plt.subplots(1,2, figsize = (8,8))

axs[0].axis('tight')

axs[0].axis('off')

the_table = axs[0].table(cellText=table_values,colWidths = [0.5], colLabels=col_labels, rowLabels = row_labels, loc='center')

the_table.set_fontsize(14)

the_table.scale(1.5, 1.5)

axs[1].pie(sizes, labels = groups, explode = explode, colors=colours, shadow=True, autopct='%1.1f%%')

plt.title('Global Proportions of 2 severely striken countries')

plt.show()