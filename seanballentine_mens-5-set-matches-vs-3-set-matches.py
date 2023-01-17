# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Data.csv', encoding = "ISO-8859-1")
#I am only working with the individual set scores so I am dropping all the irrelevent columns here
set_data = data[['ATP','Date','Tournament','W1','L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5']]

#Since n/a values are added into sets that are not reached the following code will extract all the 5 setters
five_setters = set_data.dropna(how='any')

#This section will extract from the five setters, those that are 'comebacks' in that the winner was down 1-2 after 3 sets.
comeback_bool = five_setters['W4'] > five_setters['L4']
comebacks = five_setters[comeback_bool]

#I only want the year, not the full date
comebacks.Date = comebacks.Date.str[-4:]

#result number 1
print("Of all the ", five_setters.shape[0], ' matches that went five sets during this time period ',comebacks.shape[0], " of them were 'comebacks'")
print("=======================================================")

#This section counts the number of such comebacks in the four major tournaments
comeback_by_tourn = comebacks.groupby(['Tournament', 'Date'])['ATP'].count()
comeback_by_tourn = comeback_by_tourn.unstack(level=0)
comeback_by_tourn = comeback_by_tourn[['Australian Open', 'French Open', 'Wimbledon', 'US Open']]
print(comeback_by_tourn)
print(comeback_by_tourn.sum(axis=0))
plot = comeback_by_tourn.plot()
print("========================================================")

#I also want to extract the data of matches that went 4 sets as well, since I want to know how likely it is that someone makes a 'comeback' after being down at the three set mark
fourth_set = data[['Tournament', 'Date', 'W4', 'L4']]
fourth_set.Date = fourth_set.Date.str[-4:]
four_plus = fourth_set.dropna(how='any').shape[0]
print("Of all the ", four_plus, ' matches that went four or more sets during this time period ',comebacks.shape[0], " of them were 'comebacks'")
print("=======================================================")

#Gather data for the majors again
four_plus_by_tourn = fourth_set.groupby(['Tournament','Date'])['W4'].count()
four_plus_by_tourn = four_plus_by_tourn.unstack(level=0)
four_plus_by_tourn = four_plus_by_tourn[['Australian Open', 'French Open', 'Wimbledon', 'US Open']]
print(four_plus_by_tourn)

print("=======================================================")

# This section computes the number of 3-0 matches
three_to_zero_bool = (data['W1'] > data['L1']) & (data['W2'] > data['L2']) & (data['W3'] > data['L3'])
three_to_zero = set_data[three_to_zero_bool]
three_to_zero.Date = three_to_zero.Date.str[-4:]
threes_by_tourn = three_to_zero.groupby(['Tournament','Date'])['ATP'].count()
threes_by_tourn = threes_by_tourn.unstack(level=0)
threes_by_tourn = threes_by_tourn[['Australian Open', 'French Open', 'Wimbledon', 'US Open']]

major_rates = pd.concat([threes_by_tourn.sum(axis=0), four_plus_by_tourn.sum(axis=0),comeback_by_tourn.sum(axis=0)], axis=1)
major_rates.columns = ['Three Sets', 'Four/Five Sets', 'Comebacks']
major_rates['Comeback Rate'] = major_rates['Comebacks']/major_rates['Four/Five Sets']
major_rates['Total Matches'] = major_rates['Four/Five Sets'] + major_rates['Three Sets']

from IPython.core.display import HTML
display(HTML(major_rates.to_html()))

data.Date = data.Date.str[-4:]
data2 = data.groupby(['Tournament','Date'])['ATP'].count()
data2 = data2.unstack(level=0)
data2 = data2[['Australian Open', 'French Open', 'Wimbledon', 'US Open']]
#print(data2.sum(axis=0))

#Side calculation, checking in on the missing data... looks like matches with someone retiring
inter_bool = (~(three_to_zero_bool) ) & (~ (data['W4'] > 0)) & (data['Tournament'] == 'Wimbledon')
inter = data[inter_bool]
inter = inter[['W1','L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5']]
#display(HTML(inter.to_html()))