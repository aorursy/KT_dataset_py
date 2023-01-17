# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

df=pd.read_csv('../input/malaria-dataset/estimated_numbers.csv')

df.head(3)
df.iloc[2]
df.iloc[[2,1,0]]
df[:3]
df[3:6]
df['Country'].head(3)
df.Country.head(3)
df.columns = [col.replace(' ', '_').lower() for col in df.columns]
print(df.columns)
df[['country','year']][:3]
df.year.iloc[2]
(df.country == 'Algeria').head(3)
df[df.year == 2017]
df[(df.year == 2017) | (df.who_region == 'Africa')]
df[df.country.isin(['India','Africa','Algeria'])].head()