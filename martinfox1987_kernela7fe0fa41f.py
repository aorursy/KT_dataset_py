# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

data = pd.read_csv("../input/daily-minimum-temperatures-in-me.csv", error_bad_lines=False)

data.head()

# Any results you write to the current directory are saved as output.


data.columns = ['date','temp']

data["temp"] = data["temp"].map(lambda x: x.lstrip('?'))

data["temp"] = pd.to_numeric(data["temp"])

data["date"] = pd.to_datetime(data["date"])
data.head()
import seaborn as sns

sns.set(style="darkgrid")



# Plot the responses for different events and regions

sns.lineplot(x="date", y="temp",

             data=data)