# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
global_temp = pd.read_csv('../input/GlobalTemperatures.csv')
Country = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')
City = pd.read_csv('../input/GlobalLandTemperaturesByCity.csv')
State = pd.read_csv('../input/GlobalLandTemperaturesByState.csv')
MajorCity = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')
dfs = [global_temp, Country, City, State, MajorCity]
def ends(df, x):
    return df.head(x).append(df.tail(x))
ends(global_temp,5)
yearly_avg = []
for i in range(3192//12):
    start_index = 12*i
    end_index = 12*(i+1)
    yearly_avg.append(global_temp['LandAverageTemperature'][start_index:end_index].mean())
yearly_avg = np.array(yearly_avg)

plt.figure(figsize=(20, 10))
sns.regplot(x=np.arange(1916, 2016), y=yearly_avg[-100:], order=3)
plt.show()


