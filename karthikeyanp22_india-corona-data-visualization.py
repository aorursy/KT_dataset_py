# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import date

from datetime import timedelta

from matplotlib.dates import date2num

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path=r"/kaggle/input/covid19-corona-virus-india-dataset/complete.csv"

coronaData = pd.read_csv(path,index_col=0)

coronaData.index = pd.to_datetime(coronaData.index)

coronaData.columns = ['territory', 'confirmed_cases_indians',

       'confirmed_cases_foreigners',

       'recovered', 'latitude', 'longitude', 'death']

coronaData.head()
coronaData["total_confirmed_cases"] = coronaData["confirmed_cases_indians"]+coronaData["confirmed_cases_foreigners"]

coronaData.head()
totalCasesStateWise=coronaData[coronaData.index==max(coronaData.index)].sort_values("total_confirmed_cases",ascending=False)

totalCasesStateWise.head()
fig,ax = plt.subplots(1)

fig.set_size_inches(14,6)

sns.barplot(totalCasesStateWise["territory"],totalCasesStateWise["total_confirmed_cases"])

plt.xticks(rotation=45,fontsize=10)

plt.title("Total Confirmed Cases Statewise as on "+str(date.today()),fontsize=16)

plt.xlabel("State/Union Territory",fontsize=14)

plt.ylabel("Total Confirmed Cases",fontsize=14)

plt.show()
currentCaseStats = totalCasesStateWise[["confirmed_cases_indians","confirmed_cases_foreigners","recovered","death"]].groupby(totalCasesStateWise.index).sum().transpose()

currentCaseStats.columns=["stats"]

def func(x,sum):

    return str(x.round(1))+"%"

labels=["Confirmed Cases(Indians)","Confirmed Cases(Foreigners)","Recovered","Death"]

plt.figure(figsize=(8,8))

plt.pie(currentCaseStats["stats"], labels=labels,autopct=lambda x: func(x,currentCaseStats["stats"].sum()),textprops=dict(color="w",fontsize=14),shadow=True)

plt.legend(loc="lower right",bbox_to_anchor=(1, 0, 0.4, 1),fontsize=14)

plt.title("Current Cases Stats",fontsize=16)

print(currentCaseStats)
coronaDateWise = coronaData.drop(["territory","latitude","longitude"],axis=1).groupby(coronaData.index).sum()

barwidth = 0.5

x = date2num(coronaDateWise.index)

plt.bar(x-barwidth,coronaDateWise["total_confirmed_cases"],width=barwidth,color="b")

plt.bar(x,coronaDateWise["recovered"],width=barwidth,color="r")

plt.bar(x+barwidth,coronaDateWise["death"],width=barwidth,color="g")

plt.xticks(rotation=60)
plt.bar(coronaDateWise.index,coronaDateWise["total_confirmed_cases"])

plt.xticks(rotation=60)