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
data = pd.read_csv('/kaggle/input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')
import matplotlib.pyplot as plt
import seaborn as sns
data.head(15000)
data.groupby(["Border"]).count()
data.groupby(["State"]).count()
data["Date"] = pd.to_datetime(data["Date"])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data.describe()
sns.lineplot(x = "Year", y = "Value", hue = "Border", data = data)
sns.catplot(x = 'State', y = 'Value', data = data);
sns.catplot(x="State", y="Value", hue="Border", kind="bar", data=data);
sns.barplot(x="Value", y="Measure", data=data,
            label="Total", color="b")

dv = data["Value"]
sns.boxplot(x = dv)
sns.distplot(dv, bins = 10);
sns.distplot(dv, hist = False);