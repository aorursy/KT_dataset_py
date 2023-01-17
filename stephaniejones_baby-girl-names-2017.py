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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/girls names.csv')
df.head()
columns = ['Rank', 'Name', 'Count3']
df1 = pd.DataFrame (df, columns= columns)
df1.head()
# after importing the data lets explore the data a bit more
# how many different names were given in 2017

df1.shape
# can't plot histogram as column 'count3' is objects
df1.info()
df1['Count3'] = pd.to_numeric(df1['Count3'], errors='coerce')
df1.info()
df1['Count3'].hist(bins=100)
# this histogram shows positively skewed data - the most popular names seem to be extremely popular 
# taking the log of the data will allow for more even distribution 

import numpy as np
np.log(df1)
