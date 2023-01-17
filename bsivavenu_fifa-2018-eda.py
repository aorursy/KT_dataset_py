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
df = pd.read_csv('../input/wc2018-players.csv')
df.head()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df.Team.unique()
temp = df[['Team','FIFA Popular Name','Height']].sort_values('Height',ascending=False)
print('tallest players and their team',temp.head())
print('\n')
print('shortest players and their team',temp.tail())
temp = df[['Team','FIFA Popular Name','Weight']].sort_values('Weight',ascending=False)
print('high weight players and their team',temp.head())
print('\n')
print('low weight players and their team',temp.tail())
print('total num of teams',len(df['Team'].unique()))
print(df['Team'].value_counts())
