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
import matplotlib.pyplot as plt
%matplotlib inline
# read csv file
file_path = "../input/planecrashinfo_20181121001952.csv"
data = pd.read_csv(file_path)
data.head(5)
data.shape
data.info()
data = data[["date", "fatalities", "ground"]]
data.head(5)
data.info()
data["date"].describe()
date_missing_value = data["date"][data["date"]=="?"]
len(date_missing_value)
data["fatalities"].describe()
fatalities_missing_value = data["fatalities"][data["fatalities"]=="?"]
len(fatalities_missing_value)
data["ground"].describe()
ground_missing_value = data["ground"][data["ground"]=="?"]
len(ground_missing_value)
data["year"] = data["date"].str.rsplit(",", n = 1, expand=True)[1].str.strip()
data["year"] = pd.to_numeric(data["year"], errors="coerce")
data["year"].describe()
# extract number of person in columns(aboard, fatalities, ground)
data["fatalities_num"] = data["fatalities"].str.split("(", n = 1, expand=True)[0].str.strip()
data["fatalities_num"] = pd.to_numeric(data["fatalities_num"], errors="coerce")
data["fatalities_num"].describe()
# extract number of person in columns(fatalities, ground)
data["ground_num"] = pd.to_numeric(data["ground"].str.strip(), errors="coerce")
data["ground_num"].describe()
data["total_killed"] = data["ground_num"] + data["fatalities_num"]
data["total_killed"].describe()
data.head(5)
total_killed= data[["year",  "total_killed"]].groupby("year").sum()
total_crash= data["year"].value_counts().sort_index(ascending=True).rename_axis('year').to_frame('total_crashes')
ax = total_crash.plot(figsize=(16,4))
total_killed.plot(ax=ax, secondary_y=True)