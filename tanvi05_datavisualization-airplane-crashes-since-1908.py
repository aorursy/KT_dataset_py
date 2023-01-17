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
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/airplane-crashes-since-1908/Airplane_Crashes_and_Fatalities_Since_1908.csv')

data.head(5)
data.info()
data.describe()
100 * data.isnull().sum()/len(data)

del data["Time"]
del data["Flight #"]
del data["Route"]
del data["cn/In"]
del data["Registration"]
del data["Ground"]

data.head(4)
data.dropna(subset=["Location","Operator","Type","Aboard","Fatalities"], inplace=True)
data.info()
data["Date"] = pd.to_datetime(data["Date"])
data["Date"]
100 * data.isnull().sum()/len(data)
#Let us check out surviving rate, adding a new colum to the dataset


data_copy = data.copy()
data_copy["Survival Rate"] = 100 * (data_copy["Aboard"] - data_copy["Fatalities"]) / data_copy["Aboard"]

data_copy.head(5)
data_copy_mean = data_copy["Survival Rate"].mean()
survival_per_year = data_copy[["Date","Survival Rate"]].groupby(data_copy["Date"].dt.year).agg(["mean"])
survival_per_year.plot(legend=None)
plt.ylabel("Average Survival Rate, %")
plt.xlabel("Year")
plt.title("Average Survival Rate/Year")
plt.xticks([x for x in range(1908,2009,10)], rotation='vertical')
plt.axhline(y=data_copy_mean, color='g', linestyle='--')
plt.show()
