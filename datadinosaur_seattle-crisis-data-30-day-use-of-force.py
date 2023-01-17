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
from collections import Counter
data = pd.read_csv("../input/crisis-data.csv")
cols = data.columns.tolist(); cols
"""
cols ... will remove from here the uninteresting elements

['Reported Date',
 'Reported Time',
 'Occurred Date / Time',
 'Use of Force Indicator'
"""

"exerpeince, birth, race, use of force,"
data.head()
data["Use of Force Indicator"] = data.apply(lambda x: 1 if x["Use of Force Indicator"] is "Y" else 0, axis=1)
data2 = data[['Reported Date',
 'Use of Force Indicator']]
df = data2.groupby("Reported Date").agg({np.sum}); df
df["Use of Force Indicator"]
df = df.drop("1900-01-01"); #remove Null value
chartline = df.iloc[-30:,:]
c = np.cumsum(chartline)


monthlybasine = int(np.sum(data['Use of Force Indicator'])/30); monthlybasine
baselineS = pd.Series(monthlybasine,index=c.index)

#Begin Charting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#Max Values
monthlybasine = np.array(monthlybasine)
a = max(monthlybasine,any(c.values.tolist()))
max30d = max(c.values.tolist())
maxall = max(max30d,monthlybasine)

#Initialize Graphs
plt.figure()
ax = plt.gca()
plt.plot(c,label="Daily Use of Force, Past 30")
plt.plot(baselineS,"--",label="Baseline 30-Day # of Use of Force")

#formatting
plt.ylim(0,a*1.1)
plt.xlim(c.index.values[0],c.index.values[-1])
plt.title("Running 30-Day \"Use of Force\" in Seattle's Police Dept.")
ax.legend()
tick_spacing = 7
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))


comparison = ((max30d - monthlybasine)/monthlybasine) *100
def compare():
    if comparison[0] < 0:
        return "better than"
    elif comparison[0] > 0:
        return "worse than"
    else:
        return "from"
ax.annotate("{}% {} the baseline".format(abs(comparison[0]),compare()),(15,20))

plt.show()
