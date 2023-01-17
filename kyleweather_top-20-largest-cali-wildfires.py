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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("/kaggle/input/top-20-largest-california-wildfires/top_20_CA_wildfires.csv")

data = df.copy()
data.head()
data.tail()
data.info()
data.isnull()
data['county'] = data['county'].str.replace('County','')
sns.set(style="darkgrid")

fig = plt.figure(figsize=(12,6))

ax = sns.countplot(x = 'month',data=data, order=('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'))

ax.set(xlabel='Month', ylabel='Number of fires')
fig2 = plt.figure(figsize=(12,6))

ax2 = sns.countplot(x = 'cause', data=data)

ax2.set(xlabel = 'Cause of fire', ylabel = 'Total')
sns.pairplot(data)