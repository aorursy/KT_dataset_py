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
df = pd.read_csv('/kaggle/input/mygrades/GRADES.csv')

df.head(100)
df.rename(columns={'FINAL GRADE [Total Pts: 100 Score] |547859': 'Points'}, inplace = True)
df['Points'].mean()
df['Points'].replace(0,np.nan).mean()
df['Points'].replace(0,np.nan).mean(skipna = True)
df['Points'].replace(0,np.nan).hist(bins = 28)
df[df['Points']>70].mean()
df[df['Points']>75].mean()