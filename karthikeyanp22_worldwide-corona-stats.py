# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
url="/kaggle/input/covid19-dataset/data/worldwide-aggregated.csv"

worldWide = pd.read_csv(url,index_col=0)

worldWide.index = pd.to_datetime(worldWide.index)

worldWide["Active"] = worldWide["Confirmed"] - worldWide["Recovered"] -worldWide["Deaths"]

worldWide["Recovered%"] = (worldWide["Recovered"]*100/worldWide["Confirmed"]).round(1)

worldWide["Death%"] = (worldWide["Deaths"]*100/worldWide["Confirmed"]).round(1)

worldWide["Active%"] = (worldWide["Active"]*100/worldWide["Confirmed"]).round(1)

worldWide
fig,ax = plt.subplots(1)

fig.set_size_inches(10,6)

sns.lineplot(data=worldWide["Confirmed"])

sns.lineplot(data=worldWide["Recovered"])

sns.lineplot(data=worldWide["Deaths"])

x_ticks=["2020-01-28","2020-02-13","2020-02-28","2020-03-13","2020-03-28"]

plt.xticks(x_ticks,labels=x_ticks,fontsize=12)

plt.yticks(fontsize=12)

plt.legend(["Confirmed","Recovered","Deaths"],fontsize=12)

plt.title("Total cases - Worldwide",fontsize = 15)
fig,ax = plt.subplots(1)

fig.set_size_inches(10,6)

sns.lineplot(data=worldWide["Confirmed"])

sns.lineplot(data=worldWide["Recovered"])

sns.lineplot(data=worldWide["Deaths"])

x_ticks=["2020-01-28","2020-02-13","2020-02-28","2020-03-13","2020-03-28"]

plt.xticks(x_ticks,labels=x_ticks,fontsize=12)

plt.yticks(fontsize=12)

plt.legend(["Confirmed","Recovered","Deaths"],fontsize=12)

plt.title("Total cases - Worldwide(Log Scale)",fontsize = 15)

plt.yscale("log",basey=10)
fig,ax = plt.subplots(1)

fig.set_size_inches(10,6)

sns.lineplot(data=worldWide["Active%"])

sns.lineplot(data=worldWide["Recovered%"])

sns.lineplot(data=worldWide["Death%"])

plt.xticks(x_ticks,labels=x_ticks,fontsize=12)

plt.legend(["Active%","Recovered%","Death%"],fontsize=12)

plt.title("Cases% - Worldwide",fontsize = 15)
worldWideDailyChange = worldWide.diff()

worldWideDailyChange["Death rate"] =(worldWideDailyChange["Deaths"].shift(-1)*100/worldWide["Deaths"]).round(1).shift()

worldWideDailyChange["Recovery rate"] =(worldWideDailyChange["Recovered"].shift(-1)*100/worldWide["Recovered"]).round(1).shift()

worldWideDailyChange["Increase rate"] = worldWide["Increase rate"].round(1)

worldWideDailyChange["Recovered%"] = (worldWideDailyChange["Recovered"]*100/worldWideDailyChange["Confirmed"]).round(1)

worldWideDailyChange["Death%"] = (worldWideDailyChange["Deaths"]*100/worldWideDailyChange["Confirmed"]).round(1)

worldWideDailyChange["Active%"] = (worldWideDailyChange["Active"]*100/worldWideDailyChange["Confirmed"]).round(1)

worldWideDailyChange
fig,ax = plt.subplots(1)

fig.set_size_inches(10,6)

sns.lineplot(data=worldWideDailyChange["Confirmed"])

sns.lineplot(data=worldWideDailyChange["Recovered"])

sns.lineplot(data=worldWideDailyChange["Deaths"])

plt.xticks(x_ticks,labels=x_ticks,fontsize=12)

plt.yticks(fontsize=12)

plt.legend(["Confirmed","Recovered","Deaths"],fontsize=14)

plt.title("New cases - Worldwide",fontsize = 15)
fig,ax = plt.subplots(1)

fig.set_size_inches(10,6)

sns.lineplot(data=worldWideDailyChange["Confirmed"])

sns.lineplot(data=worldWideDailyChange["Recovered"])

sns.lineplot(data=worldWideDailyChange["Deaths"])

plt.xticks(x_ticks,labels=x_ticks,fontsize=12)

plt.yticks(fontsize=12)

plt.legend(["Confirmed","Recovered","Deaths"],fontsize=14)

plt.title("New cases - Worldwide(Log Scale)",fontsize = 15)

plt.yscale("log")
fig,ax = plt.subplots(1)

fig.set_size_inches(10,6)

sns.lineplot(markers=True, dashes=False,data=worldWideDailyChange["Increase rate"])

sns.lineplot(markers=True, dashes=False,data=worldWideDailyChange["Death rate"])

sns.lineplot(markers=True, dashes=False,data=worldWideDailyChange["Recovery rate"])

plt.xticks(x_ticks,labels=x_ticks,fontsize=12)

plt.legend(["Confirmed Cases rate","Death rate","Recovery rate"],fontsize=12)

plt.title("Cases Rate - Worldwide",fontsize = 15)