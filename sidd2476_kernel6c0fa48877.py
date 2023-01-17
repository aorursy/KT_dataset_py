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
# -*- coding: utf-8 -*-

"""

Created on Thu Mar 12 13:50:23 2020



@author: SID

"""







import json

from urllib.request import urlopen



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go





# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",parse_dates=['ObservationDate'] )

data.head()



data.info()

data.isna().sum()



cases = ['Confirmed','Deaths','Recovered','Active']



data['Active'] = data['Confirmed'] - data['Deaths'] - data['Recovered']



data['COuntry/Region'] = data['Country/Region'].replace('Mainland China','China')

data[['Province/State']] = data[['Province/State']].fillna('')

data[cases] = data[cases].fillna(0)

ship = data[data['Province/State'].str.contains('Grand Princess')|data['Province/State'].str.contains('Diamond Princess cruise ship')]



china = data[data['Country/Region'] == 'China']



row = data[data['Country/Region']!='China']



new_data = data[data['ObservationDate'] == max(data['ObservationDate'])].reset_index()





china_new = new_data[new_data['Country/Region'] == 'China']

row_new = new_data[new_data['Country/Region']!='China']



data_new_group = new_data.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

china_new_group = china_new.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

row_new_group = row_new.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()





temporary = data.groupby(['Country/Region','Province/State'])['Confirmed','Deaths','Recovered','Active'].max()

temporary.style.background_gradient(cmap='Blues')
