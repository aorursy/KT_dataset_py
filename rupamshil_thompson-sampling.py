# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
dataset=pd.read_csv('/kaggle/input/ads-ctr-optimisation/Ads_CTR_Optimisation.csv')
import random

N=10000

d=10

ads_selected=[]

numbers_of_reward_0=[0]*d

numbers_of_reward_1=[0]*d

total_rewards=0

for n in range(0,N):

    ad=0

    max_random=0

    for i in range(0,d):

        random_beta=random.betavariate(numbers_of_reward_1[i]+1,numbers_of_reward_0[i]+1)

        if(random_beta>max_random):

            max_random=random_beta

            ad=i

    ads_selected.append(ad)

    rewards=dataset.values[n,ad]

    if rewards==0:

        numbers_of_reward_0[ad]+=1

    else:

        numbers_of_reward_1[ad]+=1



    total_rewards+=rewards

print(total_rewards)
plt.hist(ads_selected)

plt.show()