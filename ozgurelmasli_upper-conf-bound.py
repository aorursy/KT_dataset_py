# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/ads-ctr-optimisation/Ads_CTR_Optimisation.csv")
data.head(5)
# just choose random variable , not learning just sum of our reward value 
import random

data.describe()
# we have 10.000 data
dataCount = 10000
numberofAd = 10
sum = 0
selection = []
for item in range(0,dataCount):
    ad = random.randrange(numberofAd)
    selection.append(ad)
    val = data.values[item,ad]
    sum += val
    
    
print(sum)
plt.hist(selection)
plt.show()
import math
dataCount = 10000
d = 10
rewards = [0] * d
clickad = [0] * d

total = 0
selections = []

for item in range(1,dataCount): ## all data
    ad = 0 ## current ad
    max_ucb = 0 # max 
    for i in range(0,d): ## all cell mean ad
        if(clickad[i] > 0):
            ortalama = rewards[i] / clickad[i] # total rewar value / clickable value
            delta = math.sqrt(3/2* math.log(item)/clickad[i])
            ucb = ortalama + delta
        else :
            ucb = 10*dataCount
        if max_ucb < ucb: 
            max_ucb = ucb
            ad = i 
            
            
    selections.append(ad)
    clickad[ad] = clickad[ad]+ 1
    reward = data.values[item,ad]
    rewards[ad] = rewards[ad] + reward ## ad if 1 plus otherhande 0 
    total += reward


print(total)

plt.hist(selections)
plt.show()
import random

totalCount = 0 
selections = []
ones = [0] * d 
zeros = [0] * d 


for n in range(1,dataCount):
    ad = 0 
    max_th = 0 
    for i in range(0,d):
        betavalue = random.betavariate(ones[i] + 1 , zeros[i] + 1)
        if betavalue > max_th:
            max_th = betavalue
            ad = i 
    selections.append(ad)
    reward = data.values[n,ad]
    # take reward highest betaValue in cell
    if reward == 1:
        ones[ad] += 1
    else :
        zeros[ad] += 1
    totalCount += reward
        
print(totalCount)
plt.hist(selections)
