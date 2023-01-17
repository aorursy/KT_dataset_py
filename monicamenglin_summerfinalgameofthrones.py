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
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sn
plt.tight_layout()
df = pd.read_csv('../input/game-of-thrones-monica/character-predictions.csv',sep=",", header=0)
df.head()

#male = df[(df['male'] == 1)]
#female = df[(df['male']) != 1]
print(list(df['house']))
dfsmall = df[(df['house'] == 'House Lannister')  | (df['house'] == 'House Greyjoy') | (df['house'] =='House Stark') | (df['house'] =='House Frey') | (df['house'] =='House Targaryen')]
sn.boxplot(x="house",y="numDeadRelations",data=dfsmall)
#based on this we can see that Targaryans used to have a lot of people but now they are almost all dead compared to the other houses.
dfsmall = df[(df['house'] == 'House Lannister')  | (df['house'] == 'House Greyjoy') | (df['house'] =='House Stark') | (df['house'] =='House Frey') | (df['house'] =='House Targaryen')]
sn.violinplot(x="house",y="popularity",data=dfsmall)

#of these houses targaryan is most popular on average. But lannister has the most popular character.
#Also surprisingly lannister is more popular than stark even though stark's are the "good guys".
plt.figure(figsize=(15, 15)) # width and height in inches
co=df.corr()
sn.heatmap(co,annot=True, linewidths=1.0)
#is alive is negatively correlated with popularity
#popularity is heavily correlated with having a lot of dead relations
#so this means popular characters tend to die or have family members die
#characters in book5 is most correlated with nobles so book5 has the most nobles. being noble isn't correlated with much so maybe being a noble isn't very beneficial in GoT.
#isAliveHeir is negatively correlated with isAliveMother but positively correlated with isAliveFather, so I guess if you want your kids to be alive hopefully your dad is alive but your mother isn't. Cersei's children may be outliers causing this correlation since none have heirs but their mother is alive.

