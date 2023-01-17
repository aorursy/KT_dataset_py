# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
pd.set_option('max_columns', None)
pd.set_option('expand_frame_repr', False)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/crime.csv", index_col=0)
# print(df.head(5))

# sns.pairplot(df[['YEAR','Latitude']])
df = df[df['HUNDRED_BLOCK'] != 'OFFSET TO PROTECT PRIVACY']
print(df.head(5))
# df['HUNDRED_BLOCK'].value_counts().head(10)
# df['HUNDRED_BLOCK'].value_counts().head(20).plot.bar()
df['NEIGHBOURHOOD'].value_counts().head(20).plot.bar()

