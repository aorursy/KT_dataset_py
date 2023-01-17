# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

from matplotlib import rcParams

import missingno as msno

import seaborn as sns



# Let's ignore warnings for now

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import Datasets



df=pd.read_csv('../input/cholera-dataset/data.csv')
# Verifying



df.head()
# Fill missing values. Cases and deaths are taken as mean, while fatality rate missing values are taken as deaths/reported cases



df['Number of reported cases of cholera'] = pd.to_numeric(df['Number of reported cases of cholera'], errors='coerce')

df['Number of reported cases of cholera'] = df['Number of reported cases of cholera'].fillna(value=df['Number of reported cases of cholera'].mean())

df['Number of reported deaths from cholera'] = pd.to_numeric(df['Number of reported deaths from cholera'], errors='coerce')

df['Number of reported deaths from cholera'] = df['Number of reported deaths from cholera'].fillna(value=df['Number of reported deaths from cholera'].mean())

df['Cholera case fatality rate'] = pd.to_numeric(df['Cholera case fatality rate'], errors='coerce')

df['Cholera case fatality rate'] = df['Cholera case fatality rate'].fillna(value=df['Number of reported deaths from cholera']/

                                                                           df['Number of reported cases of cholera'])

# Find the countries where no new cholera cases have been reported since 2010 (last 10 years)

for country in np.setdiff1d(df['Country'].unique(),df[df['Year']>2010]['Country'].unique()):

    print(country)
# Seeing the trend of reported cholera cases and deaths over time



pd.pivot_table(df,index=['Year'],values=['Number of reported cases of cholera','Number of reported deaths from cholera'],aggfunc=np.sum).plot()

plt.title("Number of reported cholera cases & deaths over the years", loc='center', fontsize=12, fontweight=0, color='orange')

plt.xlabel("Year")

plt.ylabel("Total reported cholera cases and deaths")

sns.set(style="white", palette="muted", color_codes=True)

# Set up the matplotlib figure

# f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)

new_df=pd.pivot_table(df,index=['Country'],values=['Number of reported cases of cholera','Number of reported deaths from cholera'],aggfunc=np.sum)
# Plotting countries with maximum number of Cholera cases



pd.pivot_table(df,index=['Country'],values=['Number of reported cases of cholera'],aggfunc=np.sum).sort_values(by='Number of reported cases of cholera',ascending=False).head(10).plot(kind='barh')

plt.xlabel('No. of Cholera cases')
# Plotting countries with maximum number of Cholera deaths



pd.pivot_table(df,index=['Country'],values=['Number of reported deaths from cholera'],aggfunc=np.sum).sort_values(by='Number of reported deaths from cholera',ascending=False).head(10).plot(kind='barh')

plt.xlabel('No. of Cholera deaths')