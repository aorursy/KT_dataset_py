# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
plt.style.use("dark_background")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/covid19-tests-conducted-by-country/Tests_conducted_05April2020.csv')

# prepare to show by country instead of by city/region
# add a column with country name containing the sum of its cities tests
df['Country'] = df['Country or region'].apply(lambda x: x.split(':')[0])

# TODO: a precise calculation should be coutrywise instead of this mean
v = df[['Country', 'Tests/ million']].groupby(['Country'], as_index=False).mean()
v = v.sort_values(by=['Tests/ million'], ascending=False)
v.head()
# Visualize as bar chart
ax = v.plot.barh(x='Country', y='Tests/ million', fontsize=14, title='COVID-19 tests per million people', figsize=(16, 34))
ax.invert_yaxis()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(v)
# TODO: visualize historical data
df2 = pd.read_excel('/kaggle/input/covid19-tests-conducted-by-country/Tests_Conducted.xlsx')
df2.head()