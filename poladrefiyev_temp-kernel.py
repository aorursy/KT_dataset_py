# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')
df.head(10) #datanin evvelinden 10 satir getirir 
df.tail(10) #datanin sonundan 10 satir getirir. Ben sondan 10 satir 
df.info()
df.columns
fx,ax = plt.subplots(figsize=(8,8))
sns.heatmap(df.corr(), annot = True, linewidths = .5, fmt = '.1f', ax=ax)
plt.show()
df.AverageTemperature.plot(kind = 'line', color = 'red', label = 'AverageTemperature', linewidth = .5, linestyle = ':', grid = True,figsize = (18,18))
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc = 'upper right')
plt.title('Line Plot')
plt.show()
df.plot(kind = 'scatter', x = 'AverageTemperature', y = 'AverageTemperatureUncertainty', color = 'g', figsize = (18,18))
plt.show()
df.AverageTemperature.plot(kind = 'hist', bins = 50, figsize = (15,15))
plt.show()