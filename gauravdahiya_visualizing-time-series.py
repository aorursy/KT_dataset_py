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
df=pd.read_csv("../input/starbucks-data/starbucks.csv",index_col="Date",parse_dates=True)
df.head()
df.index
df.plot()
df["Close"].plot();
df["Volume"].plot();
title="TITLE"
ylabel="Y Label"
xlabel="X Label"
ax=df["Close"].plot(figsize=(20,5),title=title)
ax.autoscale(axis="both",tight=True)
ax.set(xlabel=xlabel,ylabel=ylabel)
df["Close"]["2017-01-01":"2017-12-31"].plot(figsize=(20,4))
df["Close"].plot(figsize=(20,4),xlim=["2017-01-01","2017-12-31"],ylim=[40,70],c="red")
from matplotlib import dates
df["Close"].plot(xlim=["2017-01-01","2017-03-01"],ylim=[50,60],figsize=(20,5))
ax=df["Close"].plot(xlim=["2017-01-01","2017-03-01"],ylim=[50,60],figsize=(20,5))
ax.set(xlabel="Weekly Distribution")
ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter("%a-%B-%d"))
ax=df["Close"].plot(xlim=["2017-01-01","2017-03-01"],ylim=[50,60],figsize=(20,5))
ax.set(xlabel="Weekly Distribution")

ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter("%a-%B-%d"))

ax.xaxis.set_minor_locator(dates.MonthLocator())
ax.xaxis.set_minor_formatter(dates.DateFormatter("\n\n\n\n\n%b"))
ax=df["Close"].plot(xlim=["2017-01-01","2017-03-01"],ylim=[50,60],figsize=(20,5))
ax.set(xlabel="Weekly Distribution")

ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter("%a-%B-%d"))

ax.xaxis.set_minor_locator(dates.MonthLocator())
ax.xaxis.set_minor_formatter(dates.DateFormatter("\n\n\n\n\n%b"))

ax.xaxis.grid(True)
ax.yaxis.grid(True)
