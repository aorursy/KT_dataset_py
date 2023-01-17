import csv

import matplotlib.pyplot as plot

import math

import numpy as np



all_us_stats = []

all_ru = []

all_china = []

all_india = []

all_canada = []

all_germ = []

all_uk = []

location = r'../input/all_energy_statistics.csv'

# UNITED STATES

with open(location,'r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        if(line[0] == 'United States'):

            all_us_stats.append(line)

        if(line[0] == 'Russian Federation'):

            all_ru.append(line)

        if(line[0] == 'India'):

            all_india.append(line)

        if(line[0] == 'Canada'):

            all_canada.append(line)

        if(line[0] == 'China'):

            all_china.append(line)

        if(line[0] == 'Germany'):

            all_germ.append(line)

        if(line[0] == 'United Kingdom'):

            all_uk.append(line)

household_energy_consumption_yearly = []

household_energy_consumption_amount = []

for place, type, year, measure, amount, empty, total in all_us_stats:

    if(type == "Electricity - Consumption by households"):

        household_energy_consumption_yearly.append(int(year))

        household_energy_consumption_amount.append(int(amount))

x = list(household_energy_consumption_yearly)

y = list(household_energy_consumption_amount)

xy = list(zip(x,y))

x_sorted = []

y_sorted = []





for year, amount in sorted(xy):

    x_sorted.append(year)

    y_sorted.append(math.floor(amount/1000))

# END UNITED STATES



# RUSSIA

hh_ru_amount = []

hh_ru_yearly = []

for placeru, typeru, yearru, measureru, amountru, emptyru, totalru in all_ru:

    if (typeru == "Electricity - Consumption by households"):

        hh_ru_amount.append(int(amountru))

        hh_ru_yearly.append(int(yearru))

x_ru = list(hh_ru_yearly)

y_ru = list(hh_ru_amount)

xy_ru = list(zip(x_ru,y_ru))

x_ru_sorted = []

y_ru_sorted = []



for year, amount in sorted(xy_ru):

    x_ru_sorted.append(year)

    y_ru_sorted.append(math.floor(amount/1000))

# END RUSSIA



# CANADA

hh_ca_amount = []

hh_ca_yearly = []

for placeca, typeca, yearca, measureca, amountca, emptyca, totalca in all_canada:

    if (typeca == "Electricity - Consumption by households"):

        hh_ca_amount.append(int(amountca))

        hh_ca_yearly.append(int(yearca))



x_ca = list(hh_ca_yearly)

y_ca = list(hh_ca_amount)

xy_ca = list(zip(x_ca,y_ca))



x_ca_sorted = []

y_ca_sorted = []

for year,amount in sorted(xy_ca):

    x_ca_sorted.append(year)

    y_ca_sorted.append(math.floor(amount/1000))

# END CANADA



# CHINA

hh_ch_amount = []

hh_ch_yearly = []

for placech, typech, yearch, measurech, amountch, emptych, totalch in all_china:

    if (typech == "Electricity - Consumption by households"):

        hh_ch_amount.append(int(amountch))

        hh_ch_yearly.append(int(yearch))

x_ch = list(hh_ch_yearly)

y_ch = list(hh_ch_amount)

xy_ch = list(zip(x_ch,y_ch))

x_ch_sorted = []

y_ch_sorted = []



for year, amount in sorted(xy_ch):

    x_ch_sorted.append(year)

    y_ch_sorted.append(math.floor(amount/1000))

# END CHINA



# INDIA

hh_in_amount = []

hh_in_yearly = []

for placein, typein, yearin, measurein, amountin, emptyin, totalin in all_india:

    if (typein == "Electricity - Consumption by households"):

        hh_in_amount.append(int(amountin))

        hh_in_yearly.append(int(yearin))

x_in = list(hh_in_yearly)

y_in = list(hh_in_amount)

xy_in = list(zip(x_in,y_in))

x_in_sorted = []

y_in_sorted = []



for year, amount in sorted(xy_in):

    x_in_sorted.append(year)

    y_in_sorted.append(math.floor(amount/1000))

# END INDIA



# GERMANY

hh_ge_amount = []

hh_ge_yearly = []

for placege, typege, yearge, measurege, amountge, emptyge, totalge in all_germ:

    if (typege == "Electricity - Consumption by households"):

        hh_ge_amount.append(int(amountge))

        hh_ge_yearly.append(int(yearge))

x_ge = list(hh_ge_yearly)

y_ge = list(hh_ge_amount)

xy_ge = list(zip(x_ge,y_ge))

x_ge_sorted = []

y_ge_sorted = []



for year, amount in sorted(xy_ge):

    x_ge_sorted.append(year)

    y_ge_sorted.append(math.floor(amount/1000))

# END GERMANY



# UK

hh_uk_amount = []

hh_uk_yearly = []

for placeuk, typeuk, yearuk, measureuk, amountuk, emptyuk, totaluk in all_uk:

    if (typeuk == "Electricity - Consumption by households"):

        hh_uk_amount.append(int(amountuk))

        hh_uk_yearly.append(int(yearuk))

x_uk = list(hh_uk_yearly)

y_uk = list(hh_uk_amount)

xy_uk = list(zip(x_uk,y_uk))

x_uk_sorted = []

y_uk_sorted = []



for year, amount in sorted(xy_uk):

    x_uk_sorted.append(year)

    y_uk_sorted.append(math.floor(amount/1000))

# END UK

plot.figure(figsize=(10,7))

plot.title("Total Annual Household Energy Consumption")

plot.ylabel("Thousand Megawatts-hours")

width = 0.25



usa = plot.plot(x_sorted, y_sorted, color='b', label='USA')

canada = plot.plot(x_ca_sorted, y_ca_sorted, color='c', label='CAN')

russia = plot.plot(x_ru_sorted, y_ru_sorted, color='r', label='RUS')

china = plot.plot(x_ch_sorted, y_ch_sorted, color='y', label='CHI')

india = plot.plot(x_in_sorted, y_in_sorted, color='g', label='IND')

germany = plot.plot(x_ge_sorted, y_ge_sorted, color='magenta', label='GER')

uk = plot.plot(x_uk_sorted, y_uk_sorted, color='purple', label='UK')

plot.legend(bbox_to_anchor=(1.01, 1))

plot.show()


