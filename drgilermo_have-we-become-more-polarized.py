import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.signal import savgol_filter

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



votes = pd.read_csv('../input/votes.csv')
sns.distplot(100*votes.per_point_diff_2016)

sns.distplot(100*votes.per_point_diff_2012)

plt.xlim([-100,100])

plt.xlabel('Democrats Margin [%]')

plt.title('Counties results distribution')

plt.legend(['2016','2012'], fontsize = 20)
votes['round_diff_16'] = 100*round(votes.per_point_diff_2016,2)

votes['abs_diff_16'] = np.abs(votes['round_diff_16'])

votes['round_diff_12'] = 100*round(votes.per_point_diff_2012,2)

votes['abs_diff_12'] = np.abs(votes['round_diff_12'])
plt.style.use('fivethirtyeight')

people_16 = []

people_12 = []

for difference in np.arange(0,100):

    people_16.append(np.sum(votes.total_votes_2016[votes.abs_diff_16 == difference]))

    people_12.append(np.sum(votes.total_votes_2016[votes.abs_diff_12 == difference]))

plt.plot(np.arange(0,100),people_16, 'o', alpha =0.2,  color = 'b')

yhat_16 =savgol_filter(people_16, 51, 3)

plt.plot(yhat_16,'b')

plt.plot(np.arange(0,100),people_12,'o', alpha =0.2, color = 'r')

yhat_12 =savgol_filter(people_12, 51, 3)

plt.plot(yhat_12,'r')

plt.xlim([0,70])

plt.ylim([100000,6000000])

plt.ylabel('Number of voters living in counties')

plt.xlabel('Difference between democrats and republicans (absolute value in %)')

plt.legend(['2016','2016 Running mean','2012','2012 Running mean'])

plt.title('2016 results showed more polarization')

obama = votes[votes.per_point_diff_2012>0]

romney = votes[votes.per_point_diff_2012<0]



more_dem_dem = np.true_divide(len(obama[obama.per_point_diff_2012<obama.per_point_diff_2016]), len(obama))

more_rep_dem = np.true_divide(len(obama[obama.per_point_diff_2012>obama.per_point_diff_2016]), len(obama))

flipped_dem = np.true_divide(len(obama[obama.per_point_diff_2016<0]), len(obama))



more_dem_rep = np.true_divide(len(romney[romney.per_point_diff_2012<romney.per_point_diff_2016]), len(romney))

more_rep_rep = np.true_divide(len(romney[romney.per_point_diff_2012>romney.per_point_diff_2016]), len(romney))

flipped_rep = np.true_divide(len(romney[romney.per_point_diff_2016>0]), len(romney))
plt.bar([1],[more_dem_dem], color = 'b')

plt.bar(1,more_rep_dem - flipped_dem, bottom=more_dem_dem, color = 'b', alpha = 0.3)

plt.bar(1,flipped_dem, bottom= 1 - flipped_dem, color = 'r', alpha = 0.5)



plt.bar([2],[more_rep_rep],color = 'r')

plt.bar(2,more_dem_rep - flipped_rep, bottom = more_rep_rep, color = 'r', alpha = 0.3)

plt.bar(2,flipped_rep, bottom = 1 - flipped_rep, color = 'b', alpha = 0.5)



plt.xticks([1.4,2.4], ['Democrats in 2012','Republicans in 2012'])

plt.legend(['More Polarized - Dem','Less Polarized - Dem', 'Flipped - Dem','More Polarized - Rep','Less Polarized - Rep', ' Flipped - Rep'], loc = 6)

plt.title('Republican Counties got more polarized')