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
df = pd.read_csv('../input/EconomicCalendar-2014-2018.csv',error_bad_lines=False) # raw data has a bad line, probably an extra , without quoting

df["TimeStamp"] = pd.to_datetime(df["TimeStamp"],infer_datetime_format=True,unit="s")

df.head()


df.describe(include="all")
df.to_csv("EconomicCalendar-2014-2018.csv.gz",index=False,compression="gzip")