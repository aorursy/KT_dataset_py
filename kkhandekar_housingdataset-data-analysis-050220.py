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
#importing libaries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from datetime import datetime
data = pd.read_csv("../input/housing-data-set/Housing.csv")
data.head()
data.info()
data.describe()
data.shape
data.isna().sum()
data_feature = pd.DataFrame(data, columns=['price','area','bedrooms', 'bathrooms', 'stories','parking'])

data_feature.head()
#Visualizing the Correlation

corr = data_feature.corr(method='pearson')

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(data_feature.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(data_feature.columns)

ax.set_yticklabels(data_feature.columns)

plt.show()
data.head()
#Price vs Bedrooms

sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(15,10))

ax = sns.swarmplot (x='bedrooms', y='price', data=data, hue = 'furnishingstatus')

plt.title('Price vs Bedrooms')

plt.ylabel('Price (Crores)')

plt.xlabel('Number of Bedrooms')
#Price vs Area

sns.set(style="darkgrid")

fig = plt.figure()

fig = sns.relplot(x="price", y="area", hue="furnishingstatus", data=data, kind="scatter", legend="full", height=15,aspect=1,palette="ch:r=-.5,l=.75")

fig.fig.set_size_inches(15,10)

fig.set_titles("Price vs Area")

fig.set_xlabels("Price (Crores)")

fig.set_ylabels("Area (sq. ft)")

plt.show()
#Price vs Bathrooms

sns.set(style="darkgrid")

fig = plt.figure()

fig = sns.catplot(x="bathrooms", y="price", hue="furnishingstatus", kind="bar", data=data, palette="ch:.25", height=15,aspect=1)

fig.fig.set_size_inches(15,10)

fig.set_titles("Bathrooms vs Price")

fig.set_xlabels("Bathrooms")

fig.set_ylabels("Price (Crores)")

plt.show()