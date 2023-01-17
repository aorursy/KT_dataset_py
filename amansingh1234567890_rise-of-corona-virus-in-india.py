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
#Importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
sns.set_style('whitegrid')
sns.set_context('poster')

#getting dataset
state=pd.read_csv("/kaggle/input/covid19/data_covid.csv")
testing_details=pd.read_csv("/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv")
hosp_bed=pd.read_csv("/kaggle/input/covid19-in-india/HospitalBedsIndia.csv")
state.head()

state['Date'] = state['Date'].astype('datetime64[ns]')

state["date"]=state["Date"].dt.date
confirms=state.groupby(["date"])["Confirmed"].sum().reset_index()#confirmed cases per day
deads=state.groupby(["date"])["Deaths"].sum().reset_index()#fatalities per day
cures=state.groupby(["date"])["Cured"].sum().reset_index()#recovered per day

fig = plt.figure(figsize = (20,7))

sns.lineplot(x = "date", y = "Confirmed",data=confirms ,color="blue",
             palette = "hot", dashes = False, legend="brief", label="Confirmed cases",
             linewidth=6
            )
#fig = plt.figure(figsize = (20,20))

sns.lineplot(x ="date", y = "Deaths",data=deads,
             palette = "hot", dashes = False, legend="brief", label="Deaths",
             color="red",linewidth=6
            )
sns.lineplot(x = "date", y = "Cured",data=cures, color="green",
             palette = "hot", dashes = False, legend="brief", label="Recovered",
             linewidth=6
            )
fig.autofmt_xdate()

plt.title("Corona virus in india", fontsize = 20) # for title
plt.xlabel("Date", fontsize = 20) # label for x-axis
plt.ylabel("Number of cases", fontsize = 20) # label for y-axis
plt.legend(fontsize=20)
plt.show()
fig = plt.figure(figsize = (30,15))
sns.lineplot(x = "date", y = "Confirmed",data=confirms ,color="blue",
             palette = "hot", dashes = False, legend="brief", label="Confirmed cases",
             linewidth=6
            )
sns.lineplot(x ="date", y = "Deaths",data=deads,
             palette = "hot", dashes = False, legend="brief", label="Deaths",
             color="red",linewidth=6
            )
sns.lineplot(x = "date", y = "Cured",data=cures, color="green",
             palette = "hot", dashes = False, legend="brief", label="Recovered",
             linewidth=6
            )
fig.autofmt_xdate()

plt.title("Corona virus in india", fontsize = 20) # for title
plt.yscale("log")
plt.xlabel("Date", fontsize = 20) # label for x-axis
plt.ylabel("log(Number of cases)", fontsize = 20) # label for y-axis
plt.legend(fontsize=20)
plt.show()
confirm_cases=[]
dates=[]
for i in range((len(confirms["Confirmed"])-60),len(confirms["Confirmed"])):
    confirm_cases.append(confirms["Confirmed"][i]-confirms["Confirmed"][i-1])
    dates.append(confirms["date"][i])


confirm_cases=np.array(confirm_cases)
dates=np.array(dates)

fig = plt.figure(figsize = (20,10))

sns.barplot(x =dates , y =confirm_cases ,
            palette = 'husl', edgecolor = 'w')
fig.autofmt_xdate()
plt.title("Number of confirmed cases")
plt.xticks(fontsize=8)
plt.xlabel("Date", fontsize = 15) # label for x-axis
plt.ylabel("Daily confirm cases", fontsize = 15) # label for y-axis
plt.show()

death_cases=[]
for i in range((len(confirms["Confirmed"])-60),len(confirms["Confirmed"])):
    death_cases.append(deads["Deaths"][i]-deads["Deaths"][i-1])
    

death_cases=np.array(death_cases)

fig = plt.figure(figsize = (20,10))

sns.barplot(x =dates , y =death_cases ,
            palette = 'husl', edgecolor = 'w')
fig.autofmt_xdate()
plt.xticks(fontsize=9)
plt.title("Number of fatalities")
plt.xlabel("Date", fontsize = 15) # label for x-axis
plt.ylabel("Daily death cases", fontsize = 15) # label for y-axis
plt.show()
recover_cases=[]
for i in range((len(confirms["Confirmed"])-60),len(confirms["Confirmed"])):
    recover_cases.append(cures["Cured"][i]-cures["Cured"][i-1])
    

recover_cases=np.array(recover_cases)

fig = plt.figure(figsize = (20,10))

sns.barplot(x =dates , y =recover_cases ,
            palette = 'Greens', edgecolor = 'w')
fig.autofmt_xdate()
plt.xticks(fontsize=9)
plt.title("Recovered people")
plt.xlabel("Date", fontsize = 15) # label for x-axis
plt.ylabel("Daily Recovered cases", fontsize = 15) # label for y-axis
plt.show()
confirms["active"]=confirms["Confirmed"]-(cures["Cured"]+deads["Deaths"])
# getting 7 days moving average
confirms["ma7d"]=confirms["active"].rolling(7).mean()
#confirms.tail(20)
fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(111)
ax=sns.barplot(x = "date" , y ="active" , data= confirms[-60:],
            palette = 'Reds', edgecolor = 'g')

ax2 = ax.twinx()
ax2.plot(ax.get_xticks(),confirms["ma7d"][-60:] , alpha = .75, color = 'y')
ax2.grid(False)
#sns.lineplot(x="date", y="ma7d", data=confirms[-60:],
 #           color="yellow")
fig.autofmt_xdate()
plt.xticks(fontsize=5)
plt.title("Cumulative Active cases")
plt.xlabel("Date", fontsize = 15) # label for x-axis
plt.ylabel("No. of active cases", fontsize = 15) # label for y-axis
plt.show()
fig = plt.figure(figsize = (20,10))
sns.lineplot(x = "date" , y ="active" , data= confirms,
            color = 'r', linewidth=6)
fig.autofmt_xdate()
plt.xticks(fontsize=15)
plt.yscale("log")
plt.title("Cumulative Active cases")
plt.xlabel("Date", fontsize = 15) # label for x-axis
plt.ylabel("Log(No. of active cases)", fontsize = 15) # label for y-axis
plt.show()
active= confirm_cases-(death_cases + recover_cases)
fig = plt.figure(figsize = (20,10))

sns.barplot(x =dates , y =active ,
            palette = 'YlOrRd', edgecolor = 'g')
fig.autofmt_xdate()
plt.xticks(fontsize=9)
plt.title("Active cases")
plt.xlabel("Date", fontsize = 15) # label for x-axis
plt.ylabel("Daily Active cases", fontsize = 15) # label for y-axis
plt.show()

data=state.groupby(["date"]).get_group("2020-05-26")
data=data.reset_index(drop=True)

data2=data[data.Deaths != 0].reset_index(drop=True)
sns.set_style('darkgrid')

plt.figure(figsize = (20,10))
ax=sns.barplot(x = "State/UnionTerritory", y = "Confirmed", data=data2,
            palette = 'gist_rainbow', edgecolor = 'r')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.title("Confirmed cases in different states", fontsize=20)
plt.xlabel("Name of State", fontsize = 15) # label for x-axis
plt.ylabel("No. of confirmed cases", fontsize = 15) # label for y-axis
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
plt.figure(figsize = (20,10))
ax=sns.barplot(x = "State/UnionTerritory", y = "Deaths", data=data2,
            palette = 'Reds', edgecolor = 'b')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.title("Number of Deaths in different states", fontsize=20)
plt.xlabel("Name of State", fontsize = 15) # label for x-axis
plt.ylabel("No. of confirmed cases", fontsize = 15) # label for y-axis
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
plt.figure(figsize = (20,10))
ax=sns.barplot(x = "State/UnionTerritory", y = "Cured", data=data2,
            palette = 'Greens', edgecolor = 'b')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.title("Number of Recovered cases in different states", fontsize=20)
plt.xlabel("Name of State", fontsize = 15) # label for x-axis
plt.ylabel("No. of Recovered cases", fontsize = 15) # label for y-axis
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
data2["re"] = data2["Cured"]/data2["Confirmed"]
plt.figure(figsize = (20,10))
ax=sns.barplot(x = "State/UnionTerritory", y = "re", data=data2[data2.Confirmed>=1000],
            palette = 'Greens', edgecolor = 'b')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.title("Recovery Rate", fontsize=20)
plt.xlabel("Name of State", fontsize = 15) # label for x-axis
plt.ylabel("Recovery rate", fontsize = 15) # label for y-axis
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
fig = plt.figure(figsize= (50,40))
#only those states having cases more than 1500 
data3=data2[data2.Confirmed>=1500]
data3=data3.reset_index(drop=True)
for i in range(1,len(data3["State/UnionTerritory"])):
    
    ax = fig.add_subplot(5,3,i)
    df=state.groupby("State/UnionTerritory").get_group(data3["State/UnionTerritory"][i])
    ax1=sns.lineplot(x = "date", y = "Confirmed",data=df ,color="blue",
             palette = "hot", dashes = False, legend="brief", label="Confirmed cases",
             linewidth=6
            )
    ax1.set_xticklabels(df["date"], rotation=45, ha="right")
    #fig = plt.figure(figsize = (20,20))

    ax2=sns.lineplot(x ="date", y = "Deaths",data=df,
                 palette = "hot", dashes = False, legend="brief", label="Deaths",
                 color="red",linewidth=6
                )
    ax2.set_xticklabels(df["date"], rotation=45, ha="right")
    ax3=sns.lineplot(x = "date", y = "Cured",data=df, color="green",
                 palette = "hot", dashes = False, legend="brief", label="Recovered",
                 linewidth=6
                )
    ax3.set_xticklabels(df["date"], rotation=45, ha="right")
    
    plt.title(str(data3["State/UnionTerritory"][i]), fontsize = 50) # for title
    plt.xlabel("Date", fontsize = 15) # label for x-axis
    plt.ylabel("Number of cases", fontsize = 15) # label for y-axis
    #plt.legend(fontsize=10)
    #plt.show()

plt.tight_layout(pad=3.0)    


data4=state.groupby("State/UnionTerritory")
maharast=data4.get_group("Maharashtra")
delhi=data4.get_group("Delhi")
gujarat=data4.get_group("Gujarat")
up=data4.get_group("Uttar Pradesh")
tn=data4.get_group("Tamil Nadu")
pun=data4.get_group("Punjab")

sns.set_style('darkgrid')

fig = plt.figure(figsize = (20,10))

plt.plot(delhi["date"],delhi["Confirmed"], linestyle="-",
              label="Delhi", linewidth=3
             ,color="blue"
            )
plt.plot(tn["date"], tn["Confirmed"],linestyle="-",
              label="Tamil Nadu"
             ,color="black", linewidth=3
            )
#fig = plt.figure(figsize = (20,20))

plt.plot(maharast["date"], maharast["Confirmed"],linestyle="-",
              label="Maharashtra", linewidth=3
             ,color="red"
            )

plt.plot(gujarat["date"], gujarat["Confirmed"],linestyle="-",
              label="Gujarat", linewidth=3
             ,color="violet"
            )

plt.plot(up["date"], up["Confirmed"],linestyle="-",
              label="Uttar Pradesh", linewidth=3
             ,color="green"
            )
plt.plot(pun["date"], pun["Confirmed"],linestyle="-",
              label="Punjab", linewidth=3
             ,color="yellow"
            )

fig.autofmt_xdate()

plt.title("Corona Virus in India confirmed cases(Statewise)", fontsize=30) # for title
plt.xlabel("Date", fontsize=20) # label for x-axis
plt.ylabel("Confirmed cases", fontsize=20) # label for y-axis
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.show()

fig = plt.figure(figsize = (20,10))

plt.plot(delhi["date"],delhi["Deaths"], linestyle="-",
              label="Delhi", linewidth=3
             ,color="blue"
            )
plt.plot(tn["date"], tn["Deaths"],linestyle="-",
              label="Tamil Nadu"
             ,color="black", linewidth=3
            )
#fig = plt.figure(figsize = (20,20))

plt.plot(maharast["date"], maharast["Deaths"],linestyle="-",
              label="Maharashtra", linewidth=3
             ,color="red"
            )

plt.plot(gujarat["date"], gujarat["Deaths"],linestyle="-",
              label="Gujarat", linewidth=3
             ,color="violet"
            )

plt.plot(up["date"], up["Deaths"],linestyle="-",
              label="Uttar Pradesh", linewidth=3
             ,color="green"
            )
plt.plot(pun["date"], pun["Deaths"],linestyle="-",
              label="Punjab", linewidth=3
             ,color="yellow"
            )

fig.autofmt_xdate()

plt.title("Corona Virus in India Fatalities(Statewise)", fontsize=30) # for title
plt.xlabel("Date", fontsize=20) # label for x-axis
plt.ylabel("Fatalities", fontsize=20) # label for y-axis
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.show()

labels= ['Delhi', 'Gujarat', 'Maharashtra', 'Uttar Pradesh', 'Tamil Nadu']
active_= [7300, 7200, 35000,3000,8200]
public_bed = [20572, 41129,68998 ,58310,72616]
x= np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots(figsize=(30, 15))
rects1 = ax.bar(x - width/2, active_, width, label='active_')
rects2 = ax.bar(x + width/2, public_bed, width, label='hospital beds')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('No. of beds/cases',fontsize=50)

plt.yticks(fontsize=30)
ax.set_title("Hospital beds vs Confirmed cases", fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",fontsize=30)
ax.legend(fontsize=30)

fig.tight_layout()

plt.show()
