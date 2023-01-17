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
import matplotlib.pyplot as plt

%matplotlib inline
df_ipl=pd.read_csv("../input/ipldata/matches.csv")
df_ipl.head()
df_csk=df_ipl.loc[(df_ipl['team1']=="Chennai Super Kings") | (df_ipl["team2"]=="Chennai Super Kings")]

df_csk

df_csk.shape
df_csk_won= df_csk[df_csk['winner']=='Chennai Super Kings']
df_csk_won.shape
df_csk_won['player_of_match'].value_counts().plot(kind='barh',figsize=(20,10))
df_csk_won['toss_winner'].value_counts()
df_csk_won_bat_with_toss=df_csk_won.loc[(df_csk_won["toss_decision"]=='bat') &( df_csk_won['toss_winner']=="Chennai Super Kings")]
df_csk_won_bat_with_toss.head()
df_csk_won_bat_with_toss.shape
df_csk_won['city'].value_counts()
df_csk_won['city'].value_counts().plot(kind='barh')
df_csk_won['dl_applied'].value_counts()