# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/ufcdata/data.csv")

preprocessed_data = pd.read_csv("../input/ufcdata/preprocessed_data.csv")

raw_fighter_details = pd.read_csv("../input/ufcdata/raw_fighter_details.csv")

raw_total_fight_data = pd.read_csv("../input/ufcdata/raw_total_fight_data.csv")
data=pd.read_csv('../input/ufcdata/data.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
data.B_current_lose_streak.plot(kind = 'line', color = 'g',label = 'B_current_lose_streak',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.R_win_by_Submission.plot(kind = 'line', color = 'r',label = 'R_win_by_Submission',linewidth=0.5,alpha = 0.3,grid = True,linestyle = '-')



plt.show()
data.R_win_by_Submission.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
x=data['B_current_lose_streak']>1

data[x]
data[np.logical_and(data['B_current_lose_streak']>1, data['R_Height_cms']>185)]