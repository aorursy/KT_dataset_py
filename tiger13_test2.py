# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import csv
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
beers_filepath = '/kaggle/input/craft-cans/beers.csv'
beers_by_entry_number = {}
with open(beers_filepath) as csvfile:
    csv_data = csv.reader(csvfile)
    labels = None
    for row in csv_data:
        if labels is None:
            labels = row
            continue
        entries = {}
        for entry_attribute in zip(labels, row):
            entries[entry_attribute[0]] = entry_attribute[1]
        beers_by_entry_number[row[0]] = entries
print("beer data example: {}".format(beers_by_entry_number['0']))

breweries_filepath = '/kaggle/input/craft-cans/breweries.csv'
breweries_by_id = {}
with open(breweries_filepath) as csvfile:
    csv_data = csv.reader(csvfile)
    labels = None
    for row in csv_data:
        if labels is None:
            labels = row
            labels[0] = 'brewery_id'
            continue
        entries = {}
        for entry_attribute in zip(labels, row):
            entries[entry_attribute[0]] = entry_attribute[1]
        breweries_by_id[row[0]] = entries
print("brewery data example: {}".format(breweries_by_id['0']))

lager = []
ale = []
ipa = []
others = []
for key, entry in beers_by_entry_number.items():
    if entry['ibu'] != '':
        if 'Lager' in entry['style']:
            lager.append([entry['abv'], entry['ibu']])
        elif 'Ale' in entry['style']:
            ale.append([entry['abv'], entry['ibu']])
        elif 'IPA' in entry['style']:
            ipa.append([entry['abv'], entry['ibu']])
        else:
            others.append([entry['abv'], entry['ibu']])

lager = np.array(lager)
ale = np.array(ale)
ipa = np.array(ipa)
others = np.array(others)

plt.scatter(lager[:, 0], lager[:, 1], color='r')
plt.scatter(ale[:, 0], ale[:, 1], color='b')
plt.scatter(ipa[:, 0], ipa[:, 1], color='g')
# plt.scatter(others[:, 0], others[:, 1], color='y')
plt.xlabel("abv")
plt.ylabel("ibu")

plt.show()

