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
data=pd.read_csv("/kaggle/input/startup-investments-crunchbase/investments_VC.csv",encoding = 'unicode_escape')

data.head()
data.columns
import matplotlib.pyplot as plt

import seaborn as sns
data[" market "].value_counts()[0:20]
data[data[" market "]==" Clean Technology "][0:10]
data[data[" market "]==" Education "][0:10]
data[data[" market "]==" Education "][10:17]
data[data[" market "]==" Education "][200:212]
data[" market "].value_counts()[740:]
data[data[" market "]==" Theatre "]
data[data[" market "]==" Social Opinion Platform "]
datanotnull=data.dropna()

datanotnull[datanotnull["country_code"].str.startswith("T")]
datanotnull[datanotnull["country_code"].str.startswith("t")]
data[data["country_code"]!="USA"].sort_values(by="country_code",ascending=False)[190:240]
data[data["country_code"]!="USA"].sort_values(by="country_code",ascending=False)[240:290]
data["country_code"].value_counts()[0:27]
data["country_code"].value_counts()[95:]


data.groupby("country_code").sum()["round_A"].sort_values(ascending=False)[0:14]
datanotnull["round_A"]=np.asarray(datanotnull["round_A"].astype(float))
datanotnull.groupby("country_code").sum()["round_A"].sort_values(ascending=False)[0:14]
data["round_A"]=np.asarray(data["round_A"].astype(float))

data.groupby("country_code").sum()["round_A"].sort_values(ascending=False)[0:30]
(data.groupby("country_code").sum()["round_A"].sort_values(ascending=False)[0:30]/data["country_code"].value_counts()[0:30]).sort_values()
data_only_operating=data[data["status"]=="operating"]

(data_only_operating.groupby("country_code").sum()["round_A"].sort_values(ascending=False)[0:30]/data_only_operating["country_code"].value_counts()[0:30]).sort_values()
data_only_closed=data[data["status"]=="closed"]

data_only_closed["country_code"].value_counts()
(data_only_operating["country_code"].value_counts()/data_only_closed["country_code"].value_counts()).sort_values(ascending=False)[0:40]