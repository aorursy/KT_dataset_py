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
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

train.columns
df = train[['month','day']].copy()
print('Unique values of month:',df.month.unique())

print('Unique values of day:',df.day.unique())
for column in df.columns:

    df[column].fillna(df[column].mode()[0], inplace=True)
print('Unique values of month:',df.month.unique())

print('Unique values of day:',df.day.unique())
import numpy as np



df['day_sin'] = np.sin(df.day*(2.*np.pi/7))

df['day_cos'] = np.cos(df.day*(2.*np.pi/7))

df['month_sin'] = np.sin((df.month-1)*(2.*np.pi/12))

df['month_cos'] = np.cos((df.month-1)*(2.*np.pi/12))
print(df.head(10))