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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/Fleet Data.csv')

print(data.info())
print(data.head())
alaska = data[data['Parent Airline'] == 'Alaska Airlines']

virgin = data[data['Parent Airline'] == 'Virgin America']

alaska_current = alaska[alaska['Current'] > 0]

virgin_current = virgin[virgin['Current'] > 0]

plt.subplot(1,2,1)

ak_barplot = sns.barplot(x=alaska_current['Aircraft Type'], y=alaska_current['Current'], data=alaska_current)

plt.setp(ak_barplot.get_xticklabels(), rotation=45)

_ = plt.xlabel('Aircraft Type')

_ = plt.ylabel('Currently In Service')

_ = plt.title('Alaska Airlines Fleet Composition')

plt.subplot(1,2,2)

vg_barplot = sns.barplot(x=virgin_current['Aircraft Type'], y=virgin_current['Current'], data=virgin_current)

plt.setp(vg_barplot.get_xticklabels(), rotation=45)

_ = plt.xlabel('Aircraft Type')

_ = plt.ylabel('Currently in Service')

_ = plt.title('Virgin America Fleet Composition')

plt.tight_layout()

plt.show()
frontier_legacy = data[data['Airline'] == 'Frontier Airlines']

alaska_legacy = alaska[alaska['Historic'] > 0]

plt.subplot(1,2,1)

barplot = sns.barplot(x=frontier_legacy['Aircraft Type'], y=frontier_legacy['Current'], data=frontier_legacy)

plt.setp(barplot.get_xticklabels(), rotation=45)

_ = plt.xlabel('Aircraft Type')

_ = plt.ylabel('Formerly in Service')

_ = plt.title('Legacy Frontier Fleet')

plt.subplot(1, 2, 2)

barplot = sns.barplot(x=alaska_legacy['Aircraft Type'], y=alaska_legacy['Historic'], data=alaska_legacy)

plt.setp(barplot.get_xticklabels(), rotation=45)

_ = plt.xlabel('Aircraft Type')

_ = plt.ylabel('Formerly in Service')

_ = plt.title('Legacy Alaska Fleet')

plt.tight_layout()

plt.show()
aircraft = alaska_current.groupby('Aircraft Type').sum()

current_fleet = aircraft['Current']

cost = alaska_current['Unit Cost'].str.strip('$')

cost = pd.to_numeric(cost)

cost.index = current_fleet.index

ak_cost_per_unit = current_fleet * cost

ak_total_investment = ak_cost_per_unit.sum()

v_aircraft = virgin_current.groupby('Aircraft Type').sum()

v_current_fleet = v_aircraft['Current']

v_cost = virgin_current['Unit Cost'].str.strip('$')

v_cost = pd.to_numeric(v_cost)

v_cost.index = v_current_fleet.index

v_cost_per_unit = v_current_fleet * v_cost

v_total_investment = v_cost_per_unit.sum()

print(ak_cost_per_unit)

print(ak_total_investment)

print(v_cost_per_unit)

print(v_total_investment)
ak_labels = ak_cost_per_unit.index

v_labels = v_cost_per_unit.index

plt.subplot(4,1,1)

_ = plt.pie(ak_cost_per_unit, explode=(.2,0,0,0), labels=ak_labels, autopct='%1.1f%%', shadow=True, startangle=90)

_ = plt.title('Aircraft Proportion of Total Fleet Value (Alaska Airlines)')

plt.subplot(4,1,2)

_ = plt.pie(current_fleet, explode=(.2,0,0,0), labels=ak_labels, autopct='%1.1f%%', shadow=True, startangle=90)

_ = plt.title('Number of Aircraft in Proportion to Total Fleet (Alaska Airlines)')

plt.subplot(4,1,3)

_ = plt.pie(v_cost_per_unit, explode=(.2,0), labels=v_labels, autopct='%1.1f%%', shadow=True, startangle=90)

_ = plt.title('Aircraft Proportion of Total Fleet Value (Virgin America)')

plt.subplot(4,1,4)

_ = plt.pie(v_current_fleet, explode=(.2,0), labels=v_labels, autopct='%1.1f%%', shadow=True, startangle=90)

_ = plt.title('Number of Aircraft in Proportion to Total Fleet (Virgin America)')

plt.tight_layout()

plt.show()