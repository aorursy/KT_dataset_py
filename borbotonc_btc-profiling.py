import sys
!{sys.executable} -m pip install -U pandas-profiling[notebook]
!jupyter nbextension enable --py widgetsnbextension
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport # Profiling

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_1 = "/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
file_2 = "/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"

data_coinbase = pd.read_csv(file_1)
data_bitstamp = pd.read_csv(file_2)

data_coinbase["Timestamp"] = pd.to_datetime(data_coinbase["Timestamp"], unit="s")
data_bitstamp["Timestamp"] = pd.to_datetime(data_bitstamp["Timestamp"], unit="s")
data_bitstamp.describe()
data_bitstamp.dropna(inplace=True)
profile = ProfileReport(data_bitstamp, title='Pandas Profiling Report', explorative=True)
profile.to_notebook_iframe()