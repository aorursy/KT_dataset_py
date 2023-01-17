# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # matplotlib for additional customization
import seaborn as sns # Seaborn for plotting and styling
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import dataset
battles = pd.read_csv('../input/battles.csv')
# explore dataset
battles.head()
# feature engineering
# map the attacker outcome to 1 = attacker win, 0 = attacker loss
# add another interesting feature - absolute value of attacker / defender size difference = abs(attacker_size - defender_size)
battles.attacker_outcome = battles.attacker_outcome.map({'win':1, 'loss':0})
battles['attacker_defender_size_diff'] = abs(battles['attacker_size'] - battles['defender_size'])
battles.head()
# clean up data and drop null values
battles.dropna(subset=['attacker_defender_size_diff'],inplace=True)
battles.attacker_defender_size_diff.value_counts()
# Which feature is the most important for winning a war?
# The result is super interesting - the bigger the attacker size, the lower the possibility that the attacker will win
# similar to the absolute value of the difference between the size of attacker and defender
# the bigger the difference, the lower the possibility to win 
sns.heatmap(battles.corr()[['attacker_outcome']], annot=True)
sns.set(rc={'figure.figsize':(11.7,8.27)})
# The distribution of the difference between the size of attacker and defender (absolute value)?
# For most of the war, the difference of the attacker and defender size is smaller than 5000 people.
sns.distplot(battles['attacker_defender_size_diff'], bins=20)
# Which kind of war has the biggest attacker / defender size difference (absolute value)?
# Siege has the largest difference, while ambush has almost no difference. 
sns.violinplot(battles['battle_type'], battles['attacker_defender_size_diff'])