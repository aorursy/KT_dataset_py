# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/master.csv')
data.head(5)
data.tail(5)
# print random rows from dataset

data.sample(5)
data.describe()
data.iloc[:,1:5].describe()
data.info()
data.iloc[:,1:5].info()
data.columns
#so,change the names of the column. Because there may be problems for future analysis.

data=data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})
data.columns
data.shape
data.isnull().any()
data.isnull().sum()
data = data.drop(['HDIForYear','CountryYear'], axis=1)
data.shape
min_year = min(data.Year)

max_year = max(data.Year)

print("min year : ", min_year)

print("max year : ", max_year)