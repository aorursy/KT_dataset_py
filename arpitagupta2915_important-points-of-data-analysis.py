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

df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],

                  index=['cobra', 'viper', 'sidewinder'], 

                  columns=['max_speed', 'shield'])

df
df.loc['viper']          #using index name columns are extracted
df.loc[:,'max_speed']     # all rows are extracted of column 'max_speed'
df
df.loc['cobra':'viper', 'max_speed']
df.loc['cobra', 'max_speed':'shield']
df.iloc[1:]
df.iloc[:,1]