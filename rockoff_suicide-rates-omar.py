# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = (15, 5)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
suicide = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv', encoding='latin1')

suicide
# First exploration

print('Shape:')

print(suicide.shape)

print()

print('Variable description:')

print(suicide.info())

print()

print('Head:')

print(suicide.head())

print()

print('Description:')

print(suicide.describe(include='all'))

print('Missing values:')

print()

print(suicide.isna().sum())

print()

print('Count of different values:')

for i in range(len(suicide.columns)):

    print(suicide.columns[i],':',len(suicide[suicide.columns[i]].unique()))
suicide = suicide.rename(columns = {' gdp_for_year ($) ' : 'gdp_for_year ($)'})

suicide.columns
suicide['gdp_for_year($)'] = suicide['gdp_for_year ($)'].str.replace(',', '')

suicide['gdp_for_year($)'] = suicide['gdp_for_year($)'].astype('float')

suicide['gdp_for_year($)']