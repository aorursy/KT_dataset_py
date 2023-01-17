# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter,FormatStrFormatter

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv(os.path.join(dirname, filename)) #loading the data
data.head() # quick look on data
print("US-Mexico has {} ports".format(len(data[data["Border"]=="US-Mexico Border"]["Port Name"].unique())))
print("US-Canada has {} ports".format(len(data[data["Border"]=="US-Canada Border"]["Port Name"].unique())))
plt.figure(figsize=(10,5))
plt.pie(x=[27,89],labels={"Mexico","canada"},autopct='%1.1f%%',explode=[0.05,0.05],shadow=True);
plt.title("Percentage of Ports distribution Mexico vs Canada");
from dateutil.parser import parse
data["Date"]=data["Date"].apply(lambda x: parse(x).year)
data.Measure.unique()
col=['Bus Passengers','Pedestrians','Personal Vehicle Passengers','Train Passengers']
df=data.groupby(["Date","Border","Measure"])["Value"].sum().unstack()
df[col].plot(kind="bar",stacked=True,figsize=(15,7))
plt.title("Passengers Entered across the US Borders");
plt.ylabel("No of passengers")
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d'))
df=df[col].reset_index()
df.loc[df["Border"]=="US-Canada Border",col].sum().plot(kind="bar",figsize=(15,5))
plt.title("Passengers Entered through US-canada border");
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d'))
df.loc[df["Border"]=="US-Mexico Border",col].sum().plot(kind="bar",figsize=(15,5));
plt.title("Passenegrs Entered through US-Mexico border")
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d'));
states=set(list(data["State"].values))
new={}
for i in states:
    new[i]=len(data[data["State"]==i]["Port Name"].unique())
pd.DataFrame(new.values(),new.keys(),columns={"no_of_ports"}).sort_values(by="no_of_ports").plot(kind="bar",figsize=(12,5))
plt.xticks(rotation=90)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.xlabel("States")
plt.ylabel("No of ports")
plt.title("No of Ports per State");
df2=data[data.Measure.str.contains('Bus Passengers|Pedestrians|Personal Vehicle Passengers|Train Passengers')]
df2=data.groupby(["State","Measure"])["Value"].sum().unstack()
df2[col].plot(kind="barh",figsize=(15,5),stacked=True);
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
print("Total No of Passengers entered acorss both the borders are {:,}".
      format(data[data.Measure.str.contains('Bus Passengers|Pedestrians|Personal Vehicle Passengers|Train Passengers')]["Value"].sum()))