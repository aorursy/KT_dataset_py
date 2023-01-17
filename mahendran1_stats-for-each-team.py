import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ori_data=pd.read_csv("/kaggle/input/odi-cricket-matches-19712017/originalDataset.csv")

print(ori_data.describe())
print(ori_data.head())
n=len(ori_data)

x=ori_data['Team 1']

count=0

win_count=0

team = 'India'  #Alternate the team names as[ 'India' ,'Australia', 'England','South Africa', 'New Zealand','West Indies','Sri Lanka','Pakistan' ,'Zimbabwe','Scotland','P.N.G.','Bangladesh']

y = ori_data['Team 2']

win = ori_data['Winner']

for i in range(0,n):

    if(x[i]==team and y[i]!=team):

        if(win[i]==team):

            win_count+=1

        count+=1

    if(x[i]!=team and y[i]==team):

        if(win[i]==team):

            win_count+=1

        count+=1

print("Team ",team)

print("Total number of ODI\'s : ",n)

print("ODI\'s ",team," played ",count)

print("ODI\'s that "+team+" won",win_count)

print('Win Percentage : ',round((win_count/count)*100,2))