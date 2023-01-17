# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/most_backed.csv")

dfUSD = df.loc[df['currency'] == 'usd']
num_tiers = [len(row.split()) for row in dfUSD['pledge.tier']]

dfUSD['num_tiers'] = pd.Series(num_tiers, index = dfUSD.index)
## Create an array for each number of tiers available, and fill it with amounts pledged at that number

## Would love if someone could tell me a better way to do this

pledged_by_tier = []

percent_by_tier = []

for i in range(1,101):

    pledged_by_tier.append([])

    percent_by_tier.append([])

    for index, row in dfUSD.iterrows():

        if row['num_tiers'] == i:

            pledged_by_tier[i-1].append(row['amt.pledged'])

            percent_by_tier[i-1].append(row['amt.pledged']/row['goal']*100)
#Absolute (not relative to goal)

avgs = []

medians = []

num_tiers = []

for i in range(100):

	if len(pledged_by_tier[i]) > 10:  #Ignore tier numbers for which their is insufficient data

		avgs.append(np.average(pledged_by_tier[i]))

		medians.append(np.median(pledged_by_tier[i]))

		num_tiers.append(i+1)
plt.plot(num_tiers, avgs, marker = 'o', linestyle = 'None', label = 'Average')

plt.plot(num_tiers, medians, marker = '^', linestyle = 'None', label = 'Median')

plt.legend(loc = 'upper left')

plt.xlabel('Number of Pledge Tiers Available')

plt.ylabel('Amount Pledged')

plt.show()
plt.savefig('Amount Pledged vs Number of Tiers Available.png')
#Now check the amount pledged relative to goal

avgs = []

medians = []

num_tiers = []

for i in range(100):

	if len(percent_by_tier[i]) > 10:

		avgs.append(np.average(percent_by_tier[i]))

		medians.append(np.median(percent_by_tier[i]))

		num_tiers.append(i+1)
plt.plot(num_tiers, avgs, marker = 'o', linestyle = 'None', label = 'Average')

plt.plot(num_tiers, medians, marker = '^', linestyle = 'None', label = 'Median')

plt.legend(loc = 'upper right')

plt.xlabel('Number of Pledge Tiers Available')

plt.ylabel('Percent of Goal Pledged')

plt.yscale('log')

plt.show()
plt.savefig('Percent of Goal Pledged vs Number of Tiers Available.png')