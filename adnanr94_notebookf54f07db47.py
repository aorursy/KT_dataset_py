# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
f = open('../input/births.csv', 'r')

text = f.read()
data = text.split('\n')

new_data =[]

for i in data:

    split_data = i.split(',')

    new_data.append(split_data)

print(new_data[:5])
# remove header

new_data = new_data[1:]

print(new_data[:5])
day_of_week = {}

for a in new_data:

    births = int(a[4])

    day = int(a[3])

    if day in day_of_week:

        day_of_week[day] = day_of_week[day] + births

    else:

        day_of_week[day] = births

print(day_of_week)
births_per_month = {}

for a in new_data:

    births = int(a[4])

    month = int(a[1])

    if month in births_per_month:

        births_per_month[month] = births_per_month[month] + births

    else:

        births_per_month[month] = births

print(births_per_month)
births_per_year = {}

for a in new_data:

    births = int(a[4])

    year = int(a[0])

    if year in births_per_year:

        births_per_year[year] = births_per_year[year] + births

    else:

        births_per_year[year] = births

print(births_per_year)
total_births = sum(births_per_year.values())

print(total_births)