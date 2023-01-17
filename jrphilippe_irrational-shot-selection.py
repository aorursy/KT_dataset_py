# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('fivethirtyeight')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# import data

shots = pd.read_csv('../input/shot_logs.csv', header=0)

# Any results you write to the current directory are saved as output.
shots['previous'] = np.zeros(len(shots))

shots['dist_diff'] = np.zeros(len(shots))



for i,row in enumerate(shots[1:].iterrows()):

    if i>0:

        if shots.loc[i,'GAME_ID'] == shots.loc[i-1,'GAME_ID']:

            shots.loc[i,'previous'] = shots.loc[i-1,'SHOT_RESULT']

            shots.loc[i,'dist_diff'] = shots.loc[i,'SHOT_DIST'] - shots.loc[i-1,'SHOT_DIST']

after_made = shots[shots.previous == 'made']

after_miss = shots[shots.previous =='missed']



bins = np.arange(-30,30,0.5)

x = after_made.dist_diff

y = after_miss.dist_diff



h1 = np.histogram(after_made.dist_diff,bins)

h2 = np.histogram(after_miss.dist_diff,bins)

hist_1 = np.true_divide(h1[0],sum(h1[0]))

hist_2 = np.true_divide(h2[0],sum(h2[0]))

cumu_1 = []

cumu_1.append(0)

cumu_2 = []

cumu_2.append(0)



for i,item in enumerate(hist_1):

    if i>0:

        cumu_1.append(cumu_1[i-1] + hist_1[i])

        cumu_2.append(cumu_2[i-1] + hist_2[i])

        

        

plt.plot(bins[1:]*0.3,cumu_1)

plt.plot(bins[1:]*0.3,cumu_2)

plt.legend(['After made','After miss'], loc = 2)

plt.xlabel('Difference from previous shot [m]')

plt.ylabel('Cumulative Density Function')
print('Success rate after a successful attempt...')

print(len(after_made[after_made.SHOT_RESULT == 'made'])/len(after_made))



print('Success rate after an unsuccessful attempt...')

print(len(after_miss[after_miss.SHOT_RESULT == 'made'])/len(after_miss))
print('% of 3 pointers out of all shots after a succesful attempt:')

print(len(after_made[after_made.PTS_TYPE == 3])/len(after_made))

print('% of 3 pointers out of all shots after an unsuccesful attempt:')

print(len(after_miss[after_miss.PTS_TYPE == 3])/len(after_miss))



print('% of "roughly in the paint shots" out of all shots after a succesful attempt:')

print(len(after_made[after_made.SHOT_DIST< 5])/len(after_made))

print('% of "roughly in the paint shots" out of all shots after an unsuccesful attempt:')

print(len(after_miss[after_miss.SHOT_DIST< 5])/len(after_miss))