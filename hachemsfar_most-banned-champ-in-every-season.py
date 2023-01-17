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
for j in range(3,9):
    teambans=pd.read_csv("../input/teambans.csv")
    champs=pd.read_csv("../input/champs.csv")
    matches=pd.read_csv("../input/matches.csv")
    matches=matches[matches['seasonid'] == j]
    matches.columns=['matchid', 'gameid', 'platformid', 'queueid', 'seasonid', 'duration',
           'creation', 'version']
    df=pd.merge(matches,teambans,how='inner',on='matchid')
    champs.columns=['champion_name', 'championid']
    df1=pd.merge(df,champs,how='inner',on='championid')
    df2=set(df1['champion_name'])
    df3=list(df2)
    max=0
    for i in df3:
        if max<df1[df1['champion_name']==i].shape[0]:
            max=df1[df1['champion_name']==i].shape[0]
            champ_name=i
            ratio_ban=(max/df1.shape[0])*100
    print("Season "+ str(j))
    print("Most Banned Champ: "+ champ_name)
    print("% that champ will be banned: "+ str(ratio_ban))
