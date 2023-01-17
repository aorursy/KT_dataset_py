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
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
df
df.describe()
df.groupby("Last Update")["Confirmed"].value_counts()
df["Last Update"] = pd.to_datetime(df["Last Update"])
df["date"] = df["Last Update"].dt.date
g = df.groupby("date")["Confirmed"].sum().reset_index()
g.plot()
g["shift"] = g["Confirmed"].diff()

g
g["shift"].plot()