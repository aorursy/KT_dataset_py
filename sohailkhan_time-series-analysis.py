# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

sns.set(rc={"figure.figsize": (6, 6)})
sns.set_style('whitegrid')

# Any results you write to the current directory are saved as output.
global_temps_file = '../input/GlobalTemperatures.csv'

GlobalTemps = pd.read_csv(global_temps_file)
GlobalTemps.head()
def makeTimeSeries(df):
    ts = pd.to_datetime(df.dt)
    df.index = ts
    return df.drop('dt', axis=1)
ts = makeTimeSeries(GlobalTemps)
ts.LandAverageTemperature.plot()
pd.rolling_mean(ts.LandAverageTemperature, 10, freq='A').plot()
# zoom in to that time frame and take a rolling min, max, and mean of the time
yearWithoutSummer = ts['1800':'1850'].LandAverageTemperature
pd.rolling_min(yearWithoutSummer, 24).plot()
pd.rolling_max(yearWithoutSummer, 24).plot()
pd.rolling_mean(yearWithoutSummer, 24).plot()
batches = 100

min_year = ts.index.min().year
max_year = ts.index.max().year

for t in range(min_year, max_year - batches, batches):
    beginning = str(t)
    end       = str(t+batches)
    data = ts[beginning:end].LandAverageTemperature
    data.plot(kind='kde',alpha=0.7, label='{}-{}'.format(beginning, end))
plt.legend(loc='best')