# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
countries = pd.read_csv('/kaggle/input/undata-country-profiles/country_profile_variables.csv',usecols = [0,1,2,3,4,5,6,7,8])

countries.columns= countries.columns.str.replace('(','',regex=True)

countries.columns= countries.columns.str.replace(')','',regex=True)

countries.columns= countries.columns.str.replace(',','',regex=True)

countries.columns= countries.columns.str.replace('%','',regex=True)

countries.columns= countries.columns.str.replace('.','',regex=True)

countries.columns= countries.columns.str.replace('$','D',regex=True)

countries.columns= countries.columns.str.replace(' ','_',regex=True)

countries.head(8)
countries.tail(10)
countries.shape[0]
len(countries.columns)
countries = countries.set_index('country')

countries.head()
countries.loc['Indonesia','Population_in_thousands_2017']
countries.Region.iloc[-10:]
countries[countries.Region == 'South-easternAsia']
countries.query('Population_in_thousands_2017 >= 100000').shape[0]
countries.groupby('Region')['Population_in_thousands_2017'].sum().sort_values(ascending=False).head(10)
countries['Sex_ratio_m_per_100_f_2017'].mean()
countries['Sex_ratio_m_per_100_f_2017'].idxmin()
countries.groupby('Region')['Surface_area_km2'].size().plot.pie()
countries.GDP_per_capita_current_USD.sort_values().tail(20).plot.barh()
countries.groupby('Region')['Sex_ratio_m_per_100_f_2017'].mean().plot.bar(legend=True)