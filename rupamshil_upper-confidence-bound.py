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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset=pd.read_csv('/kaggle/input/ads-ctr-optimisation/Ads_CTR_Optimisation.csv')
import math #to perform math operations like logarithm

N=10000 #no of times the ads are shown

d=10 #number of ads

ads_selected=[] #ads that are selected will be append to this list

number_of_selection=[0]*d #how many times each add is selected 

sums_of_rewards=[0]*d #no of rewards each ads got

total_rewards=0 #total no of rewards

for n in range(0,N):

    ad=0 #selected ad is initialised to 0

    max_upper_bound=0 #maximum upper nond is initialised to zero

    for i in range(0,d): #seceond for loop to take each ad in the row n

        if(number_of_selection[i]>0):   #as first number_of_selection all the values are intialized to zero and something by zero doesnt make sense

            average_rewards=sums_of_rewards[i]/number_of_selection[i]

            

            delta_i =math.sqrt(((3/2)*math.log(n+1))/number_of_selection[i])

            

            upper_bound=average_rewards+delta_i

        else:

            

            upper_bound=1e400

        if(upper_bound>max_upper_bound):

            

            max_upper_bound=upper_bound

            ad=i

            

            

    ads_selected.append(ad)

    number_of_selection[ad]+=1

    rewards=dataset.values[n,ad]

    sums_of_rewards[ad]+=rewards

    total_rewards+=rewards

    

    

print(ads_selected)

print(number_of_selection)

print(sums_of_rewards)

print(total_rewards)

            

            

       

        
plt.hist(ads_selected)

plt.title('Histogram of ads selections')

plt.xlabel('Ads')

plt.ylabel('Number of times each ad was selected')

plt.show()