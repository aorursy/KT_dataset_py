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
import matplotlib.pyplot as plt
nCov_data_org = pd.read_csv("/kaggle/input/covid19-coronavirus/2019_nCoV_data.csv", index_col="Sno")

nCov_data_org["Date"] = pd.to_datetime(nCov_data_org["Date"])

nCov_data_org["Last Update"] = pd.to_datetime(nCov_data_org["Last Update"])

nCov_data_org.head()
nCov_data_org["Date"].value_counts()
nCov_data = nCov_data_org[["Date", "Confirmed", "Deaths", "Recovered"]]

nCov_data.head()
nCov_Date = nCov_data.groupby(["Date"]).sum()

nCov_Date.head()
plt.figure(figsize=(12,6), dpi=80)

plt.plot(nCov_Date, marker='o')

plt.legend(nCov_Date.columns)

plt.xticks(rotation=60)

plt.show()
nCov_Date["Mortality"] = nCov_Date["Deaths"]/nCov_Date["Confirmed"]

plt.figure(figsize=(12,6), dpi=80)

plt.plot(nCov_Date["Mortality"]*100, marker='o')

plt.ylabel("Mortality [%]")

plt.xticks(rotation=60)

plt.show()
nCov_motarity = nCov_data_org
plt.figure(figsize=(12,6), dpi=80)

nCov_Date["Mortality"] = nCov_Date["Deaths"]/nCov_Date["Confirmed"]

plt.figure(figsize=(12,6), dpi=80)

plt.plot(nCov_Date["Mortality"]*100, marker='o', label="World")



nCov_motarity = nCov_data_org

_data_china = nCov_motarity[nCov_motarity["Country"] == "Mainland China"].groupby("Date").sum()

_data_others = nCov_motarity[nCov_motarity["Country"] != "Mainland China"].groupby("Date").sum()

_data_china["Motarity"] = _data_china["Deaths"]/_data_china["Confirmed"]

_data_others["Motarity"] = _data_others["Deaths"]/_data_others["Confirmed"]

plt.plot(_data_china["Motarity"]*100, marker='o', label="Mainland China")

plt.plot(_data_others["Motarity"]*100, marker='o', label="Others")



plt.xticks(rotation=60)

plt.legend()

plt.ylabel("Motarity [%]")

plt.show()
nCov_Latest = nCov_data_org[:-1]

nCov_Latest = nCov_Latest[["Country", "Confirmed", "Deaths", "Recovered"]]

nCov_Country = nCov_Latest.groupby(["Country"]).sum()

nCov_Country.head()
def plot_stacked_bar_chart(data):

    fig = plt.figure(figsize=(8, 8), dpi=80)

    ax = fig.add_axes((0, 0, 1, 1),

                      ylabel=data.index.name,

                      xlabel="")

    bottom = 0

    for i in range(0, 2):

        ax.barh(data.index, data.iloc[:, i], label=data.columns[i],

                color="C"+str(i), 

                #zorder=10-0.1*i,

                left=bottom)

        #bottom += data.iloc[:, i]

    ax.legend(title=data.columns.name,

              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    ax.grid(True, axis="x", color="gainsboro", alpha=0.8, zorder=7)

    #ax.set_xticks(rotation=60)

    [ax.spines[side].set_visible(False) for side in ["right", "top"]]

    
plot_stacked_bar_chart(nCov_Country[nCov_Country.index!="Mainland China"].sort_values("Confirmed"))