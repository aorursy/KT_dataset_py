# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#using a file from our dataset pane
df = pd.read_csv("../input/athlete_events.csv",sep=",",header =0)
print(list(df.columns.values))
print(list(df['Team']))
df.head()
dfsmall = df[(df['Team'] == 'China') | (df['Team'] == 'United States') | (df['Team'] == 'Finland') | (df['Team'] == 'Norway') | (df['Team'] == 'Turkey') | (df['Team'] == 'Pakistan') | (df['Team'] == 'Edgyt') | (df['Team'] == 'Singapore')]
sn.boxplot(x="Team",y="Height",data=dfsmall)
#china's average height is a bit low but they have the tallest person among these countries. It also has the shortest person and is the most diverse.
dfsmall = df[(df['Team'] == 'China') | (df['Team'] == 'United States') | (df['Team'] == 'Finland') | (df['Team'] == 'Norway') | (df['Team'] == 'Turkey') | (df['Team'] == 'Pakistan') | (df['Team'] == 'Edgyt') | (df['Team'] == 'Singapore')]
sn.violinplot(x="Team",y="Weight",data=dfsmall)
#Norway has the highest average weight, but USA has the heaviest person of these countries. China has the lightest person.
co= df.corr()
sn.heatmap(co,annot=True, linewidths=1.0)
#weight and height have a strong correlation
#age is most correlated with weight, probably because of metabolisms slowing.
#year isn't strongly correlated with other measurements which means that over time athletes are about the same in terms of height, weight and the composition of athletes hasn't changed over time.