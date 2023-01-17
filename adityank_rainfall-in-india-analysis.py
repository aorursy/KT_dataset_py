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
import json

import datetime

import ast

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#importing dataset (csv file)

data = pd.read_csv('../input/rainfall-in-india/rainfall in india 1901-2015.csv')
overall = data.groupby(by='SUBDIVISION').sum()[['ANNUAL']].sort_values(by='ANNUAL', ascending=False).head(10)

overall
high_rain = data[data['ANNUAL'].notnull()][['SUBDIVISION', 'YEAR','ANNUAL']].sort_values('ANNUAL', ascending=False).head(10)

high_rain.reset_index(inplace=True)

high_rain
low_rain = data[data['ANNUAL'].notnull()][['SUBDIVISION', 'YEAR','ANNUAL']].sort_values('ANNUAL', ascending=False).tail(10)

low_rain.reset_index(inplace=True)

low_rain
karnataka = data[data['SUBDIVISION'] == 'COASTAL KARNATAKA'][['YEAR','ANNUAL']]



karnataka.plot(figsize=(18,5),x='YEAR',y='ANNUAL')
data.groupby("YEAR").sum()['ANNUAL'].plot(figsize=(12,8));
data[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',

       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("YEAR").sum().plot(figsize=(13,8));
data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',

       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").mean().plot.barh(stacked=True,figsize=(13,8));