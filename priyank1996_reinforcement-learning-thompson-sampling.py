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
ads_df = pd.read_csv("../input/Ads_Optimisation.csv")

ads_df.head()
ads_df.info()
import matplotlib.pyplot as plt

import seaborn as sns

ads = ads_df.columns

ad_reward = [ads_df.values[:,i].sum() for i in range(len(ads))]

plt.xticks(rotation = 45)

sns.barplot(ads, ad_reward, palette='rainbow')

plt.ylabel('Performance')

plt.xlabel('Ad_Type')

plt.title('Ads_Performance')
import random

d = len(ads)

rounds = ads_df.values.shape[0]

ad_selected = []

total_reward = 0



for n in range(rounds):

    ad = random.randint(1,10)

    reward = ads_df.values[n,ad-1]

    total_reward += reward

    ad_selected.append(ad-1)



print('Total Reward on Random Selection of Ads: {}'.format(total_reward)) 
sns.countplot(ad_selected, palette='RdBu_r')

plt.xlabel('Ad_Type')

num_ad_i_selcted =[0]*d   # where each column represents each ad

total_reward_i = [0]*d    # columns will represent the clicks each ad got

ad_selected = []

total_reward = 0

for n in range(rounds):

    max_ucb = 0

    ad= 0

    for i in range(0,d):

        '''this if-else condition makes sure that for the first 10 rounds we have each ad selected once

        so that we have some values to start with'''

        if num_ad_i_selcted[i]>0:

            average_reward = total_reward_i[i]/num_ad_i_selcted[i] # Calculating average reward

            delta_i = np.sqrt((3/2)*np.log(n+1)/num_ad_i_selcted[i])

            ucb = average_reward + delta_i # Defining the upper bound

        else:

            ucb = 1e400

        if ucb>max_ucb:

            max_ucb = ucb

            ad = i

    ad_selected.append(ad)

    num_ad_i_selcted[ad] += 1

    reward = ads_df.values[n,ad]

    total_reward_i[ad] += reward

    total_reward += reward



print('Total Clicks on Ads: {}'.format(total_reward))
plt.xticks(rotation = 45)

sns.barplot(ads, num_ad_i_selcted)
d = 10

rounds = 10000

num_reward_0 =[0]*d # where each column represents each ad

num_reward_1 = [0]*d

ad_selected = []

total_reward = 0

for n in range(rounds):

    max_random = 0

    ad= 0

    for i in range(0,d):

        random_dist = random.betavariate(num_reward_1[i]+1,num_reward_0[i]+1 )

        if random_dist>max_random:

            max_random = random_dist

            ad = i

    ad_selected.append(ad)

    reward = ads_df.values[n,ad]

    if reward == 1:

        num_reward_1[ad] += 1

    else:

        num_reward_0[ad] += 0

    total_reward += reward



print(total_reward)
plt.xticks(rotation = 45)

sns.countplot(ad_selected)