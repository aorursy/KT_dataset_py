# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import csv
import itertools

data = csv.reader(open("../input/NFA 2018.csv"))

headers = list(itertools.islice(data, 1))[0]
# get list of countries
countries = []
countryIdx = headers.index('country')

for row in data:
    country = row[countryIdx]
    if country not in countries:
        countries.append(country)
    countries
# get map: <Country>: [<year_data>, <year_data>, <year_data>...]

data = csv.reader(open("../input/NFA 2018.csv"))

data_by_country = {}
countryIdx = headers.index('country')

for row in data:
    country = row[countryIdx]
    if country == 'country':
        next
    
    if country not in data_by_country:
        data_by_country[country] = []
    data_by_country[country].append(row)
# Get highest GDP for each country

