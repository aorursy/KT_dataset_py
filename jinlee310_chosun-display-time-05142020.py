# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
my_data=pd.read_csv("../input/chosun-time-05122020/Dynamodb_chosun_time05-12-2020.csv")
my_data.head()
my_data.describe(include="all")
my_nzDTdata=my_data[my_data['Disconnect Time']>0]
my_nzDTdata.describe(include="all")
import matplotlib.pyplot as plt

DisconnectTime=my_nzDTdata["Disconnect Time"]
plt.hist(DisconnectTime, bins=1000)
plt.hist(DisconnectTime,range=(0, 3000), bins=1000)
my_nzCTdata=my_data[my_data['Click Time']>0]
my_nzCTdata.describe(include="all")
ClickTime=my_nzCTdata["Click Time"]
plt.hist(ClickTime, range=(0,700), bins=100)
ser = DisconnectTime.to_numpy() # type(ser) numpy.ndarray
for i in range(1, 25):
    count[i] = np.count_nonzero(  (300 * (i-1) < ser) &  (ser <= 300 * i) )
count_series = pd.Series(count)
count_nz=count_series[1:]
count_nz.plot(kind='bar')
count_series.plot(kind='bar')