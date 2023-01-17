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
!wget https://s3.amazonaws.com/imcbucket/data/flights/2008.csv
import pandas as pd
df = pd.read_csv('2008.csv')
df.shape
print(df.columns)
df.head
df2 = df[['UniqueCarrier','ArrDelay']]
df2.head
df3 = df2.groupby(['UniqueCarrier']).mean()
df3.head
df3 = df3.sort_values(by=['ArrDelay'])

# df3 = df3.sort_values(by=['ArrDelay'], ascending=False)
df3.head()
df3.shape
import matplotlib.pyplot as plt

plt.figure()

df3.plot.bar()

# plt.axhline(0, color='k')