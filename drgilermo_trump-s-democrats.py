# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline
#Read Data

votes = pd.read_csv('../input/votes.csv')
for i,row in enumerate(votes.iterrows()):

    if row[1].votes_gop_2016>row[1].votes_dem_2016:

        votes.loc[i,'winner_16'] = 'Trump'

    else:

        votes.loc[i,'winner_16'] = 'Clinton'  

    if row[1].votes_gop_2012>row[1].votes_dem_2012:

        votes.loc[i,'winner_12'] = 'Romney'

    else:

        votes.loc[i,'winner_12'] = 'Obama' 
flipped = votes[votes.winner_16 == 'Trump']

flipped = flipped[flipped.winner_12 == 'Obama']
flipped['adj_2012_dem_votes'] = flipped.votes_dem_2012*(1+flipped.population_change/100)

flipped['adj_2012_gop_votes'] = flipped.votes_gop_2012*(1+flipped.population_change/100)
plt.figure(figsize = (8,8))

plt.plot(flipped.adj_2012_dem_votes,flipped.votes_dem_2016,'o', alpha = 0.6)

plt.plot(flipped.adj_2012_gop_votes,flipped.votes_gop_2016,'o',alpha = 0.6)

plt.plot([0,350000],[0,350000])

plt.xlim([0,75000])

plt.ylim([0,75000])

plt.xlabel('Adjusted number of votes - 2012')

plt.ylabel('Number of votes - 2016')

plt.legend(['Dem','Rep','2012 = 2016'], loc = 2, numpoints = 1)

plt.title('Trumps and Clintons performance compered to Romney and Obama')
plt.figure(figsize = (8,8))

plt.plot(flipped.adj_2012_dem_votes,flipped.votes_dem_2016,'o', alpha = 0.6)

plt.plot(flipped.adj_2012_gop_votes,flipped.votes_gop_2016,'o',alpha = 0.6)

plt.plot([0,350000],[0,350000])

plt.xlim([0,20000])

plt.ylim([0,20000])

plt.xlabel('Adjusted number of votes - 2012')

plt.ylabel('Number of votes - 2016')

plt.legend(['Dem','Rep','2012 = 2016'], loc = 2, numpoints = 1)

plt.title('Trumps and Clintons performance compered to Romney and Obama')
x = flipped['state_abbr'].value_counts().index.tolist()

states = flipped['state_abbr'].value_counts()

all_counties =  votes['state_abbr'].value_counts()
plt.figure(figsize = (9,9))

plt.bar(range(len(states)),states)

plt.xticks(range(len(x)), x, size='small')

plt.xlabel('State')

plt.ylabel('Number of counties flipped')
all_counties =  votes['state_abbr'].value_counts()

new_dict = dict()

for key in states.keys():

    temp = np.true_divide(states[key],all_counties[key])

    new_dict[key] = temp

 

l = sorted(new_dict, key=new_dict.get)

y = []

for state in l:

    y.append(new_dict[state])
plt.figure(figsize = (9,9))

plt.bar(range(len(y)),y)

plt.xticks(range(len(l)), l, size='small')

plt.xlabel('State')

plt.ylabel('% of counties flipped')
clinton_flipped = votes[votes.winner_16 == 'Clinton']

clinton_flipped = clinton_flipped[clinton_flipped.winner_12 == 'Romney']

print('Number of counties flipped by Clinton:',len(clinton_flipped))

print('Number of counties flipped by Trump:',len(flipped))

print('Total number of counties:', len(votes))