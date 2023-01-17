import matplotlib.pyplot as plt

import csv

import numpy as np



dates = []

confirmed = []

deaths = []



with open('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv') as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')

    daysSince = 0

    for row in csv_reader:

        if row[2] == "Anhui":

            dates.append(daysSince)

            confirmed.append(float(row[5]))

            deaths.append(float(row[6]))

            daysSince += 1

  

# plotting the points  

plt.plot(dates, confirmed, label="confirmed")

plt.plot(dates, deaths, label="deaths")

plt.xticks(rotation=90)

plt.xlabel("Days since 22nd January")

plt.ylabel("Number")

plt.title("Coronavirus cases in Anhui province, China.")

plt.legend()

plt.show()