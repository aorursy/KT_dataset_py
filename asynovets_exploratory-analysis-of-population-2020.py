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
df = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

df.head()
# df['Med. Age'] = df['Med. Age'].astype('int64')

df['Med. Age'].value_counts()
x = df[df['Med. Age'] != 'N.A.']

x['Med. Age'].describe()

# df['Med. Age'].value_counts()
x['Med. Age'].astype('int64')
x['Med. Age'].median(), round(x['Med. Age'].mean(),1)
import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(10,6))

sns.distplot(x['Med. Age'], bins = 30)

plt.show()