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
df= pd.read_csv("/kaggle/input/covid19-bangladesh/covid.csv",parse_dates=['Date'])
df[['Date','New Cases']].tail()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
%matplotlib inline
plt.rcParams.update({'font.size': 14})

myFmt = mdates.DateFormatter('%b-%d')

fig, ax = plt.subplots(figsize=(11, 6))

Date=df['Date']
NR=df['New Recoveries']
NC=df['New Cases']
ND=df['New Deaths']

ax.xaxis.set_major_formatter(myFmt)
ax.plot(Date,NR,linewidth=2,color="green",label="Recoveries")
ax.plot(Date,ND,linewidth=2,color="red",label="Deaths")
ax.plot(Date,NC,linewidth=2,color="orange",label="Cases")
ax.set_xlim([datetime.date(2020, 4, 1), datetime.date(2020, 4, 30)])
plt.xticks(rotation=45)
plt.grid(color='#a1d1d0')
plt.legend()
#plt.savefig("chart1.png", bbox_inches="tight", pad_inches=1)
plt.show()
plt.close()
active_cases=df['Total Cases'].max()-df['Total Deaths'].max()-df['Total Recoveries'].max()
exp_vals = [active_cases,df['Total Deaths'].max(),df['Total Recoveries'].max()]
exp_labels = ["Active Cases","Deaths","Recovered"]
plt.axis("equal")
plt.pie(exp_vals,labels=exp_labels, shadow=True,colors=["orange","red","green"],
        autopct='%1.1f%%',radius=1.5,explode=[0,0.1,0.2],counterclock=True, startangle=45)

plt.savefig("piechart.png", bbox_inches="tight", pad_inches=1, transparent=True)
plt.show()
plt.close()
fig= plt.figure(figsize=(11,6))
df2=df[df['Date']>'2020-03-15']
xpos = np.arange(len(df2))
xpos

Date=[date_obj.strftime('%b-%d') for date_obj in df2['Date']]

NR=df2['New Recoveries']
ND=df2['New Deaths']


plt.bar(xpos-0.2,NR,width=0.4,color="green",label="Recoveries")
plt.bar(xpos+0.2,ND,width=0.4,color="red",label="Deaths")

plt.xticks(xpos,Date,rotation=90)
plt.title("Deaths vs. Recoveries")
plt.grid()
plt.legend()
plt.savefig("chart2.png", bbox_inches="tight", pad_inches=1)
plt.show()
plt.close()
plt.axis("equal")
plt.pie([df['Total Deaths'].max(),df['Total Recoveries'].max()],labels=["Deaths","Recoveries"], shadow=True,colors=["red","green"],
        autopct='%1.1f%%',radius=1.5,explode=[0,0.1],counterclock=True, startangle=45)

plt.savefig("piechart.png", bbox_inches="tight", pad_inches=1, transparent=True)
plt.show()
plt.close()
