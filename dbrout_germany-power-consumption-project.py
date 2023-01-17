import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!ls ../input
dailydata = pd.read_csv('../input/opsd_germany_daily.csv')

dailydata.shape
dailydata.head(10)
dailydata.tail(10)