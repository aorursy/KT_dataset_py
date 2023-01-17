# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dt=pd.read_csv("../input/corona-virus-report/country_wise_latest.csv")
dt.head()
dt.head(10)
dt.info()
dt.corr()
dt.Recovered.plot(kind="line",color="red",figsize=(20,10),label="Recovered",grid=True,linewidth=1)
plt.title("Recovered statistics ditribution by countries ")
plt.xlabel("Countries")
plt.ylabel("Recovered")
plt.show()

ab=dt[(dt["Deaths"]<2000)]
bc=dt[(dt["Confirmed"]<2000)]
ab.Deaths.plot(kind="line",linewidth=1)
bc.Confirmed.plot(kind="line",alpha=0.5,grid=True)
plt.xlabel("Countries")
plt.ylabel("Number")
plt.show()
a=dt[np.logical_and(dt['Deaths']<2000, dt['Recovered']<2000 )]
a.Recovered.plot()
plt.show()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(dt.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()