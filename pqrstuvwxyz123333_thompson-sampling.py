



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import random

import matplotlib.pyplot as plt

data=pd.read_csv("../input/Ads_CTR_Optimisation.csv")

data.describe()
d=10

row=0

column=0

delta_i=0

max_delta_i=0

ad=0

ad_selected=[]

N=10000

reward=0

total_reward=0
number_of_ad_with_1=[0]*d

number_of_ad_with_0=[0]*d
for row in range(0,N):

    ad=0

    max_delta_i=0

    for column in range(0,d):

        delta_i=random.betavariate(number_of_ad_with_1[column]+1,number_of_ad_with_0[column]+1)

        if(delta_i>max_delta_i):

            max_delta_i=delta_i

            ad=column

    ad_selected.append(ad) 

    if(data.values[row,ad]==1):

            number_of_ad_with_1[ad]=number_of_ad_with_1[ad]+1

    else:

            number_of_ad_with_0[ad]=number_of_ad_with_0[ad]+1

    reward=data.values[row,ad]

    total_reward=total_reward+reward
ad_selected
plt.hist(ad_selected)
total_reward