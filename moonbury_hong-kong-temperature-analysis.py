# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

data = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')

import warnings

warnings.filterwarnings('ignore')
data = data.dropna()

hkdata=data[data['Country']=='Hong Kong']

hkdata.dt = pd.to_datetime(hkdata.dt, format='%Y-%m-%d')

hkdata['Year'] = hkdata.dt.apply(lambda x: x.year)

hkdata['month'] = hkdata.dt.apply(lambda x: x.month)

print(hkdata.head())

print(hkdata.shape[0])
#hkdata.plot('dt','AverageTemperature')



p = sb.stripplot(data=hkdata, x='Year', y='AverageTemperature');

p.set(title='Hong Kong Temperature from 1840')

dec_ticks = [y if not x%20 else '' for x,y in enumerate(p.get_xticklabels())]

p.set(xticklabels=dec_ticks);
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(hkdata.month[0:12],hkdata.AverageTemperature[0:12])

ax.plot(hkdata.month[13:24],hkdata.AverageTemperature[13:24])