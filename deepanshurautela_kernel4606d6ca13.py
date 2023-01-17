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
from datetime import datetime, timedelta



today = datetime.now() - timedelta(days=1)

yesterday = datetime.now() - timedelta(days=2)



today_strfmt = today.strftime("%Y-%m-%d")

yesterday_strfmt = yesterday.strftime("%Y-%m-%d")



print(today_strfmt, yesterday_strfmt)
df = pd.read_csv("https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv")

#Dropping the co-ordinates columns

df = df.drop(["Lat", "Long"],axis="columns")

#Setting the index to all three to use state data as well

df = df.set_index(["Date","Country/Region", "Province/State"])

print(df.head())
df = df.query(f"Date == '{today_strfmt}' or Date == '{yesterday_strfmt}'")

df.to_csv('data.csv')

print(df.tail(200))