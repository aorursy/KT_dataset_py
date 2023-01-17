# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math as m



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.describe()
df.head()
# Useful functions

def make_bins_simple(min_bin, max_bin, size):

    '''

    Create bins for histogram plots

    '''

    return list(range(min_bin, max_bin + size, size))





def make_bins_max(df_col, size):

    '''

    Create bins for histogram plots

    '''

    min_val = m.floor(min(df_col)/10)*10

    max_val = m.ceil(max(df_col)/10)*10

    

    return list(range(min_val, max_val + size, size))





def age_sex_df(df, min_age, max_age, sex):

    '''

    Custom filter data frame

    '''

    return df.loc[(df.age > min_age) & (df.age < max_age) & (df.sex == sex)]
male_df = age_sex_df(df, 0, 100, 1)

female_df = age_sex_df(df, 0, 100, 0)
bins = make_bins_max(df.age, 5)

sns.distplot(male_df.age, bins = bins, label = 'male')

sns.distplot(female_df.age, bins = bins, label = 'female')

plt.legend()

plt.show()
sns.kdeplot.__code__.co_varnames
f = (df.loc[(df.trestbps > 0) & (df.chol > 0) & (df.target == 0)])

m = (df.loc[(df.trestbps > 0) & (df.chol > 0) & (df.target == 1)])



sns.kdeplot(f.trestbps, f.chol, label = 'female')

sns.kdeplot(m.trestbps, m.chol, label = 'male')

plt.xlabel('resting blood pressure (mmHg)')

plt.ylabel('serum cholesterol (mg/dl)')

plt.legend()

plt.show()
plt.scatter(df.trestbps, df.target)
no_presence = (df.loc[(df.trestbps > 0) & (df.target == 0)]).trestbps

presence = (df.loc[(df.trestbps > 0) & (df.target == 1)]).trestbps

bins = make_bins_max(df.trestbps, 10)

sns.distplot(no_presence, label = 'no presence', bins = bins)

sns.distplot(presence, label = 'presence', bins = bins)

plt.xlabel('resting blood pressure (mmHg)')

plt.ylabel('frequency')

plt.legend()

plt.show()