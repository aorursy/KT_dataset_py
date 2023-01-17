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
import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

%matplotlib inline
data_path = Path('/kaggle/input/accidental-drug-related-deaths-20122018/')

data = data_path / 'Accidental_Drug_Related_Deaths_2012-2018.csv'
df_data = pd.read_csv(data)
df_data.head(5)
df_data.columns
df_data.shape
fig, axs = plt.subplots(figsize=(23,6))

df_data['Age'].value_counts().plot(kind='bar')
fig, axs = plt.subplots(figsize=(23,6))

df_data['ResidenceState'].value_counts().plot(kind='bar', log=True)
fig, axs = plt.subplots(figsize=(23,6))

df_data['MannerofDeath'].value_counts().plot(kind='bar', log=True)
fig, axs = plt.subplots(1,4, figsize=(23,6))



df_data['Age'].hist(bins=90, ax=axs[0])

df_data['Sex'].value_counts().plot(kind='bar', legend=True, ax=axs[1])

df_data['Race'].value_counts().plot(kind='bar', legend=True, ax=axs[2])

df_data['Location'].value_counts().plot(kind='bar', legend=True, ax=axs[3])



plt.savefig('data_exploration_plot.png',dpi=300)



plt.show()