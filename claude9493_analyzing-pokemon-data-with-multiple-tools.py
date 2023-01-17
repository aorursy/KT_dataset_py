# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Pokemon.csv', index_col='Name')
print(data.head(10)) 
# Some id appears several times, this column is useless.
data = data.drop(['#'], axis=1)
# Rewrite the columns' name in uppercase
data.columns = data.columns.str.upper()
data.columns
# All the types
types = pd.Series(data["TYPE 1"].unique())
print(data.info())
# data.info shows that there are missing value in the column 'TYPE 2'
# which means that nearly half of all Pokemon have only one type
typ1 = data['TYPE 1'].unique()
typ2 = pd.Series(data['TYPE 2'].unique()).drop(1) # drom nan
Double_types = pd.DataFrame([[x,y] for x in typ1 for y in typ2 if x != y], columns=['TYPE 1', 'TYPE 2']) 
# Use a list comprehensions to generate all combinations of different types.
Double_types_stat = [data[(data["TYPE 1"] == Double_types.iloc[i][0]) & (data["TYPE 2"] == Double_types.iloc[i][1])].shape[0] for i in range(0, 306)]
Double_types['AMOUNT'] = Double_types_stat
# For each combination of types, record how many pokemon have such kind of types
pt = Double_types.pivot_table(index='TYPE 1', columns='TYPE 2', values='AMOUNT', aggfunc=np.sum)
pt = pt.fillna(0)
# Make the previous dataframe to a table, element is the number of pokemon with row and column 's type.
f, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(pt,  annot=True, linewidths=0.1, ax = ax, cmap="YlGnBu")
# use seaborn.heatmap to visualize it
sns.jointplot(x = 'ATTACK', y = 'DEFENSE', data=data, kind = 'hex')
sns.jointplot(x = 'SP. ATK', y = 'SP. DEF', data=data, kind = 'hex')