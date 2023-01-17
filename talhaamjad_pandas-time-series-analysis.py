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
df = pd.read_csv("../input/appledata/aapl.csv" ,parse_dates=["Date"], index_col="Date")

df.head()
df.index
df['2017-06-30']
df["2017-01"]
df['2017-06'].head()
df['2017-06'].Close.mean()
df['2017'].head(2)
df['2017-01-08':'2017-01-03']
df['2017-01']
df['Close'].resample('M').mean().head()
df['2016-07']
%matplotlib inline

df['Close'].plot()
df['Close'].resample('M').mean().plot(kind='bar')
df['2017-01-08':'2017-01-03']
df['Close'].resample('Y').mean().plot(kind='bar')