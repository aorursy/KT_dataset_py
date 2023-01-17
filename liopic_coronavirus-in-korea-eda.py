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
df = pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')

df.head()
df.columns
df.sex.value_counts()
df.region.value_counts()
grouped = df.groupby('region').mean()

grouped.birth_year
df.group.value_counts()
church_members = df[df.group == 'Shincheonji Church']

church_members.head()
church_members.describe()
df.groupby('confirmed_date').count()
import matplotlib.pyplot as plt
count_church = church_members.groupby('confirmed_date').count()['id']

count_church
count_not_church = df[df.group != 'Shincheonji Church'].groupby('confirmed_date').count()['id']

count_not_church
joined = pd.concat([count_church,count_not_church], axis = 1).fillna(0)

joined.columns = ['church', 'no_church']

joined.head()
df.sample(10)
joined.plot.bar(figsize=(20,5))

plt.yscale('log')