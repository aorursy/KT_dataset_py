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
df = pd.read_csv('../input/videogamesales/vgsales.csv')
df.head(3)
df.iloc[2]
df.loc[2]
df.loc[[3, 4, 5]]
df.iloc[[2, 1, 3]] 
df[:3]
df[3:6]
df['Genre'].head(3)
df.columns = [col.replace(' ', '_').lower() for col in df.columns]

print(df.columns)
df[['platform', 'year']][:3]
df.platform.iloc[2]
df.platform.iloc[[2]]
(df.platform == 'NES').head(3)
df[df.platform == 'NES']
df.head()
df[df['name'].str.split().apply(lambda x: len(x) == 3)].head(3)
df[df.platform.isin(['NES', 'Wii', 'GB'])].head()