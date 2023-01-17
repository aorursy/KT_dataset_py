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
county = pd.read_csv('/kaggle/input/coronavirus-covid19-data-in-the-united-states/us-counties.csv')
ny = county[county.county=='New York City']
ny['NewCases']=ny['cases']-ny['cases'].shift(1)
ny['NewDeaths']=ny['deaths']-ny['deaths'].shift(1)
ny.set_index(keys = ny.date, drop = True, inplace = True)
ny.NewDeaths.plot(kind = 'bar')
ny.NewCases.plot(kind = 'bar')
state = pd.read_csv('/kaggle/input/coronavirus-covid19-data-in-the-united-states/us-states.csv')
nys = state[state.state == 'New York']
nys.reset_index(drop =True,inplace=True)
nys['NewCases']=nys['cases']-nys['cases'].shift(1)
nys['NewDeaths']=nys['deaths']-nys['deaths'].shift(1)
nys.set_index(keys = nys.date, drop = True, inplace = True)
nys.NewDeaths.plot(kind = 'bar')
nys.NewCases.plot(kind = 'bar')
import matplotlib.pyplot as plt

plt.figure(figsize=plt.figaspect(0.2))

plt.plot(nys.NewCases)
plt.plot(ny.NewCases)
plt.plot(nys.NewDeaths)
plt.plot(ny.NewDeaths)



