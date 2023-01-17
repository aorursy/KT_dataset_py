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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt   

import seaborn as sns

%matplotlib inline
import pandas as pd

filename = "../input/covid19-global-forecasting-week-1/train.csv"

dt = pd.read_csv(filename)

t = dt.describe()

print(t)
from pandas import read_csv

from matplotlib import pyplot

series = read_csv('../input/covid19-global-forecasting-week-1/train.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

series.plot()

pyplot.show()
import pandas as pd

filename = "../input/hubei-china-data/Hubei_China_2.csv"

df = pd.read_csv(filename)

c = df.describe()

print(c)
China_Hubei = pd.read_csv('../input/hubei-china-data/Hubei_China_2.csv')

China_Hubei.head()

China_Hubei.info()

China_Hubei.describe()

China_Hubei.columns
sns.pairplot(China_Hubei)
sns.distplot(China_Hubei['ConfirmedCases'])
sns.distplot(China_Hubei['Fatalities'])
from pandas import read_csv

from matplotlib import pyplot

series = read_csv('../input/hubei-china-data/Hubei_China_2.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

series.plot()

pyplot.show()
import pandas as pd

filename = "../input/daytoday-china-hubei/DayToDay.csv"

df = pd.read_csv(filename)
from pandas import read_csv

from matplotlib import pyplot

series = read_csv('../input/daytoday-china-hubei/DayToDay.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

series.plot()

pyplot.show()
import pandas as pd

filename = "../input/submissioncsv/submission.csv"

df = pd.read_csv(filename)



s = df.describe()

print(s)
Submission = pd.read_csv('../input/submissioncsv/submission.csv')

Submission.head()

Submission.info()

Submission.describe()

Submission.columns
from pandas import read_csv

from matplotlib import pyplot

series = read_csv('../input/submissioncsv/submission.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

series.plot()

pyplot.show()
from pandas import read_csv

from matplotlib import pyplot

from pandas.plotting import lag_plot

series = read_csv('../input/submissioncsv/submission.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

lag_plot(series)
Submission.to_csv('submission.csv',index=False)