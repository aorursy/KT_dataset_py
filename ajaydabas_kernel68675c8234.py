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
df=pd.read_csv('../input/fifa19/data.csv')
df.describe()
df.head(3)
df.iloc[1]
df.loc[99]
df.loc[[7,9,77,99]]
df.iloc[[9,15,0]]
df[:9]
df[9:19]
df['Nationality'].head(5)
df.Nationality.head(9)
df.columns = [col.replace(' ', '_').lower() for col in df.columns]

print(df.columns)
df[['name', 'position']][:8]
df.name.iloc[0]
df.name.iloc[[0]]
(df.nationality == 'Argentina').head(9)
df[df.nationality == 'Argentina']
df[(df.stamina > 90) | (df.ballcontrol > 75)].head(5)
df[df['name'].str.split().apply(lambda x: len(x) == 1)].head(5)
df[df.club.isin(['FC Barcelona', 'Real Madrid', 'Juventus'])].head(10)