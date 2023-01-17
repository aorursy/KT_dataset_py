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
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
data.head(60)
data['time'] = pd.to_datetime(data['Last Update'])
import seaborn as sns

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
ax = sns.lineplot(x="time", y="Confirmed", data=data)
ax = sns.lineplot(x="time", y="Deaths", data=data)
ax = sns.distplot(data['Confirmed'])
ax = sns.distplot(data['Deaths'])
ax = sns.distplot(data['Confirmed'].cumsum())
ax = sns.distplot(data['Deaths'].cumsum())
data['Confirmed'].cumsum().plot()
data['Deaths'].cumsum().plot()