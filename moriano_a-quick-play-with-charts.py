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
raw = pd.read_csv("../input/lasvegas_tripadvisor.csv")
raw.head(n=2)
raw.dtypes
raw.info()
raw.describe()
def plot_by(df, column_name, size=(20, 15), sortit=True, horizontal=True):
    by_user_country = df.copy()
    x = by_user_country.groupby(column_name)["Nr. reviews"].agg(sum)
    kind = "barh" if horizontal else "bar"
    if sortit:
        x.sort_values().plot(kind=kind, figsize=size, title="By " + column_name)
    else:
        x.plot(kind=kind, figsize=size, title="By " + column_name)
plot_by(raw, "User continent", size=(10, 5))
plot_by(raw, "User country", size=(15, 12))
plot_by(raw, "Traveler type", size=(10, 5))
plot_by(raw, "Review weekday", size=(10, 5), sortit=False, horizontal=False)
plot_by(raw, "Score", size=(10, 5), horizontal=False)
plot_by(raw[raw["User country"] == "Australia"].sort_values(by="Score"), "Score", size=(10, 5), horizontal=False)
raw.groupby("Hotel name")
