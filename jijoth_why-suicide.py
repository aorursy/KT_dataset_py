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
df=pd.read_csv("/kaggle/input/suicides-in-india/Suicides in India 2001-2012.csv")
import seaborn as sns

import matplotlib.pyplot as plt

df.columns
df=df[df['State']!="Total (All India)"]

df=df[df['State']!="Total (States)"]

df=df[df['State']!="Total (Uts)"]
state_total=df.groupby(['State'])['Total'].sum().reset_index()
state_total
plt.figure(figsize=(16, 10))

sns.barplot(x='Total',y='State',data=state_total.sort_values(by='Total',ascending=False))
year_total=df.groupby(['Year'])['Total'].sum().reset_index()
plt.figure(figsize=(16, 10))

sns.barplot(x='Year',y='Total',data=year_total.sort_values(by='Year',ascending=True))