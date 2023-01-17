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
ipldf = pd.read_csv('../input/matches.csv')

print(ipldf.head(5))
print(ipldf.describe())
print(ipldf.columns)

print(ipldf.values)
print(ipldf.groupby('season').size())

print(ipldf.groupby('season').size().plot(kind='bar'))
idf = pd.DataFrame(ipldf.groupby('toss_winner').size())

# print(idf.columns)

print(idf.plot(kind='bar'))
idf2 = ipldf.groupby('winner').size().plot(kind='bar')

print(idf2)
idf = ipldf.groupby('player_of_match').size().to_frame('size')

print(idf.sort_values(by='size', ascending=False).head(10).plot(kind='bar', figsize=(15,5)))