# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import os

import datetime as dt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.



main_dataset = pd.read_csv(r'/kaggle/input/us-accidents/US_Accidents_Dec19.csv')

main_dataset.shape
main_dataset.dtypes
main_dataset.describe()
corr = main_dataset.corr(method='pearson') 

fig, ax = plt.subplots(figsize=(9,9)) 

sb.heatmap(corr, ax = ax,cmap='coolwarm',  robust=True)

ax.set_title('Correlation')

plt.show()
mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(9,9)) 

cmap = sb.diverging_palette(240, 10, as_cmap=True)

sb.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0, ax=ax)

ax.set_title('Correlation')

plt.show()
main_dataset['date'] = main_dataset['Start_Time'].str.split(n=1).str[0]

main_dataset['Date'] = pd.to_datetime(main_dataset['date'], errors='coerce')

main_dataset['Week'] = main_dataset['Date'] .dt.week

main_dataset['Year'] = main_dataset['Date'] .dt.year
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(15,7))

main_dataset.groupby(['Year','Week','Severity']).count()['ID'].unstack().plot(ax=ax)

ax.set_xlabel('Week')

ax.set_ylabel('Number of Accidents')
main_dataset['timestamp'] = pd.to_datetime(main_dataset['Weather_Timestamp'], errors='coerce')

main_dataset['Hour'] = main_dataset['timestamp'] .dt.hour

main_dataset['Minute'] = main_dataset['timestamp'] .dt.minute

hours = [hour for hour, df in main_dataset.groupby('Hour')]

plt.plot(hours, main_dataset.groupby(['Hour'])['ID'].count())

plt.xticks(hours)

plt.xlabel('Hour')

plt.ylabel('Numer of accidents')

plt.grid(True)

plt.show()