# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
total_fights = pd.read_csv('/kaggle/input/ufcdata/raw_total_fight_data.csv', sep= ';')

total_fights.head()
#creating new DataFrame with winner of each fight and win type



winner_type = total_fights[['Winner', 'win_by']]

winner_type
#Check if there are NaN values in DF



winner_type.info()
#all ways to win. We need only 'KO/TKO' and "TKO - Doctor's Stoppage"

winner_type.win_by.unique()
#There are no KO or TKO wins in NaN values, so we can drop them.



nan_winners = winner_type[winner_type.Winner.isna()]

nan_winners.win_by.unique()
winner_type = winner_type.dropna()

winner_type.head()
ko_dict = {}



for index, info in winner_type.iterrows():

    if info['Winner'] not in ko_dict:

        if 'KO' in info['win_by']:

            ko_dict.update({info['Winner'] : 1})

    elif info['Winner'] in ko_dict:

        if 'KO' in info['win_by']:

            ko_dict[info['Winner']] += 1
sorted_d = sorted(ko_dict.items(), key=lambda x: x[1], reverse= True)

sorted_d
#Vitor Belfort has most KO/TKO wins in UFC history until 2019.