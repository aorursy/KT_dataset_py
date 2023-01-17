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
confirmed = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')
confirmed.head()
confirmed.loc[confirmed['Country/Region']=='Canada']
canada = confirmed.loc[confirmed['Country/Region']=='Canada'].iloc[:,4:].sum(axis=0)
canada.tail()
canada.plot(label='Canada')
plt.legend()
plt.title("Number of confirmed cases in Canada")
plt.show()
countries=['Brazil','Italy','Germany','Canada','Australia']
for c in countries:
    confirmed.loc[confirmed['Country/Region']==c].iloc[:,4:].sum(axis=0).plot(label=c)
plt.legend()
plt.title('Total Number of COVID-19 Confirmed Cases')
plt.xlabel('Day')
plt.ylabel('Number of Cases')