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
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df['Country/Region'].values
import matplotlib.pyplot as plt
data = {}



for i, row in df.iterrows():

    _temp = np.array([], dtype=np.int)

    _new = np.array([], dtype=np.int)

    for col in range(4, df.shape[1]-7, 7):

        _temp_tot = 0

        _temp_new = 0

        for j in range(7):

            _temp_tot += df.iloc[i, col+j]

        _temp_new += df.iloc[i,col+7]-df.iloc[i,col]

        _temp = np.append(_temp, _temp_tot)

        _new = np.array([0], dtype=np.int) if col == 4 else np.append(_new, _temp_new)

    label = row['Country/Region']

    _ref = label if label != 'Korea, South' else 'South Korea'

    if row['Country/Region'] in data.keys():

        data[_ref]['Total'] = np.add(data[label]['Total'], _temp)

        data[_ref]['New'] = np.add(data[label]['New'], _new)

    else:

        data[_ref] = {}

        data[_ref]['Total'] = _temp

        data[_ref]['New'] = _new



_all_tot = np.zeros(len(data[list(data.keys())[0]]['Total']), dtype=np.int)

_all_new = np.zeros(len(data[list(data.keys())[0]]['New']), dtype=np.int)



for key in data:

    _all_tot = np.add(_all_tot, data[key]['Total'])

    _all_new = np.add(_all_new, data[key]['New'])



data['All'] = {'Total' : _all_tot, 'New' : _all_new}
DISPLAYED_COUNTRIES = ['United Kingdom', 'US', 'Australia', 'Canada', 'Russia', 'China', 'South Korea', 'Spain', 'Italy', 'Germany', 'Japan', 'France']
fig = plt.figure()

fig.set_size_inches(18.5, 10.5, forward=True)

plt.ylabel('New Cases per 7 Days')

plt.xlabel('Total Number of Cases')

legend_labels = []

ax = fig.add_subplot(1,1,1)

ax.loglog()

for frame in data:

    if frame in DISPLAYED_COUNTRIES:

        legend_labels.append(frame)        

        ax.plot(data[frame]['Total'], data[frame]['New'], '-' if frame in 'United Kingdom' else 'x')

ax.legend(legend_labels)

plt.show()