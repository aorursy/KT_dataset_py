import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random

import math
data=pd.read_csv("../input/Ads_CTR_Optimisation.csv")
N=10000

row=0

column=0

d=10

ads_selected=[]

No_of_Ad_selected=[0]*d

toltal_reward_of_ad=[0]*d

totalreward=0

ad=0

reward=0

delta_column=0
for row in range(0,N):

    max_upper_Bounds=0

    ad=0

    reward=0

    for column in range(0,d):

        if(No_of_Ad_selected[column]>0):

            average_reward=toltal_reward_of_ad[column]/No_of_Ad_selected[column]

            delta_column=math.sqrt(3/2 * math.log(row+1) / No_of_Ad_selected[column])

            upper_bounds=average_reward+delta_column

        else:

            upper_bounds=1e400

        if(upper_bounds>max_upper_Bounds):

            max_upper_Bounds=upper_bounds

            ad=column

    ads_selected.append(ad)

    No_of_Ad_selected[ad]=No_of_Ad_selected[ad]+1

    reward=data.values[row,ad]

    toltal_reward_of_ad[ad]=toltal_reward_of_ad[ad]+reward

    totalreward=totalreward+reward

    

