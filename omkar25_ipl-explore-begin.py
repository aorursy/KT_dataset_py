# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
deliveries=pd.read_csv("../input/deliveries.csv")

matches=pd.read_csv("../input/matches.csv")
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',

                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',

                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors']

                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'],inplace=True)
years=matches.season.unique()

for i in years:

    win=matches[matches['season']==i].winner.values[len(matches[matches['season']==i])-1]

    print ("Winner of IPL tournament held in",i,"was",win)
from collections import Counter

win_till_group=[]

x=len(matches[matches['season']==2008])-3

for i,j in zip(matches['winner'],range(x)):

    win_till_group.append(i)

score_table=Counter(win_till_group)

score_table
import operator

summ=sorted(score_table.items(),key=operator.itemgetter(1),reverse=True)

score_table=dict(summ)

print("================================")

print("| Team |Won|Lost|Total_points|")

for i in score_table:

    print(i,score_table[i],(14-score_table[i]),(score_table[i]*2))

print("================================")

#need to improve presentation