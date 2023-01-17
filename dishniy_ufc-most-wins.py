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
data = pd.read_csv('/kaggle/input/ufcdata/raw_total_fight_data.csv', sep=';')

data.head()
winner = data[['Winner']]

winner = winner.dropna()

winner
win_dict = {}



for i in winner.Winner:

    if i not in win_dict:

        win_dict.update({i : 1})

    else:

        win_dict[i] +=1
sorted_d = sorted(win_dict.items(), key=lambda x: x[1], reverse= True)

sorted_d
# Donal Cerrone has most wins in UFC.