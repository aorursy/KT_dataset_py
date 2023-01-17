# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data.columns
data.head()
data.drop(["Photo","Flag","Club Logo","Unnamed: 0"],axis=1,inplace=True)
data  = data.loc[:,["Name","Age",'Nationality',"Overall","Potential","Club","Special","Preferred Foot","Wage","Value",
                    "International Reputation","Weak Foot",'Position',"Jersey Number"]]
data.info()
data.describe()
#Age Destribution of players
plt.hist(data.Age)
plt.show()
def value_extract(values):
    value = values.replace("€","")
    if 'M' in value:
        value = float(value.replace("M",""))*100000
    elif "K" in value:
        value = float(value.replace("K",""))*1000
    return float(value)
data.Value = data.Value.apply(lambda x : value_extract(x))
data.Wage = data.Wage.apply(lambda x : value_extract(x))
data.Nationality.value_counts()[:10].plot(kind='bar',figsize=(8,6))
wage = data.groupby(['Overall'])["Wage"].mean()
value = data.groupby(["Overall"])["Value"].mean()

wage = wage.apply(lambda x : x/1000)
value = value.apply(lambda x : x/1000000)

data["Wage by Potential"] = data["Wage"]
data["Value by Potential"] = data["Value"]

wage_p = data.groupby("Potential")["Wage by Potential"].mean()
value_p = data.groupby("Potential")["Value by Potential"].mean()

wage_p = wage_p.apply(lambda x : x/1000)
value_p = value_p.apply(lambda x : x/1000000)

fife_overall = pd.concat([wage,value,wage_p,value_p],axis=1)
fife_overall.plot(figsize=(8,8))
# Wage and Value by age
wage_value = data.groupby("Age")[["Wage","Value"]].mean()#.plot(figsize=(10,8))
#rint(wage_value)
wage_value.Wage = wage_value.Wage.apply(lambda x : x/1000)
wage_value.Value = wage_value.Value.apply(lambda x : x/1000000)
wage_value.plot(figsize=(10,8))
growth = data.groupby('Age')["Overall"].mean()
potential = data.groupby('Age')["Potential"].mean()

summ = pd.concat([growth,potential],axis=1)
summ.plot()
plt.legend(loc='best')
# We can see that left preferred foot player get more wage
wa_val = data.groupby("Preferred Foot")[["Wage","Value"]].mean().reset_index()
print(wa_val)
width = 0.5
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.bar(np.arange(len(wa_val)),np.log(wa_val.Wage),0.25,label="Wage",color="r")
plt.bar(np.arange(len(wa_val))+.25,np.log(wa_val.Value),0.25,label = 'Value',color='b')
plt.legend(loc='best')
plt.xticks(np.arange(2),["Left","Right"])

plt.subplot(1,2,2)
plt.bar(np.arange(len(wa_val)),np.log(wa_val.Wage),0.25,label="Wage",color="r")
plt.bar(np.arange(len(wa_val))+.25,np.log(wa_val.Value),0.25,label = 'Value',color='b')
plt.legend(loc='best')
plt.xticks(np.arange(2),["Left","Right"])
plt.tight_layout()
#Expensive Player
data[data.Wage == data.Wage.max()]
#Let's find the clubs with most player rating over 85
players = data[data.Overall > 85]
players = players.Club.value_counts()
plt.figure(figsize=(12,10))
sns.barplot(x=players.index,y=players.values)
plt.xticks(rotation=90)
plt.xlabel("Club")
plt.ylabel("No. of Players(Over 85)")
