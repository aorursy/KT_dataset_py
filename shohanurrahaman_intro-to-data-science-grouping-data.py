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
import pandas as pd 

df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

display(df.head())
cols = ['ocean_proximity', 'population']

filterd_df = df[cols]

total = filterd_df.groupby('ocean_proximity').sum()

print(total)
import pandas as pd 

df = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-mat.csv')

display(df.head())
cols = ['school', 'sex', 'G3']

new_df = df[cols]

print(new_df.head())



grp = new_df.groupby(['school', 'sex']).mean()

print(grp)
