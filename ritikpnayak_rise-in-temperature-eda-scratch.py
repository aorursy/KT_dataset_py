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
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')
df.head()
df.shape
df['Region'].value_counts()
df.info()
plt.figure(figsize=(15,8))

sns.set_style('darkgrid')



sns.lineplot(x = df.loc[df['Region']=='Asia','Year'], y = df.loc[df['Region']=='Asia','AvgTemperature'], lw=8, color = 'red')
plt.figure(figsize=(15,8))



sns.distplot(df['AvgTemperature'],

            kde_kws = {'color':'orange','lw':6},

            hist_kws = {'color':'red','lw':6})
plt.figure(figsize=(15,8))



sns.kdeplot(df['AvgTemperature'], shade=True)
sns.jointplot(x = df.loc[df['Region']=='Asia','Year'], y = df.loc[df['Region']=='Asia','AvgTemperature'], kind = 'reg')
plt.figure(figsize=(15,8))



sns.barplot(x=df.loc[df['Region']=='Asia','Year'], y=df.loc[df['Region']=='Asia','AvgTemperature'])
plt.figure(figsize=(15,8))



sns.scatterplot(x=df.loc[df['Region']=='Asia','Year'], y=df.loc[df['Region']=='Asia','AvgTemperature'], hue=df.loc[df['Region']=='Asia','City'])
plt.figure(figsize=(15,8))



sns.regplot(x=df.loc[df['Region']=='Asia','Year'], y=df.loc[df['Region']=='Asia','AvgTemperature'])