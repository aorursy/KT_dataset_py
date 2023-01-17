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
# Used World Happiness Report 2017

year_2017=pd.read_csv('../input/2017.csv')
year_2017.info()
year_2017.shape
year_2017.head(10)
year_2017_melted=pd.melt(frame=year_2017,id_vars="Country",value_vars=["Happiness.Score","Economy..GDP.per.Capita.","Family","Health..Life.Expectancy.","Freedom","Generosity","Trust..Government.Corruption.","Happiness.Rank"])
year_2017_melted
year_2017_new=year_2017_melted.pivot(index="Country",columns="variable",values="value")
year_2017_new
year_2017_last=year_2017_new.sort_values(by="Happiness.Rank",ascending=True)
year_2017_last
year_2017_last.rename(columns={"Happiness.Score":"Happiness_Score","Economy..GDP.per.Capita.":"Economy_GDP_per_Capita",
                               "Health..Life.Expectancy.":"Health_Life_Expectancy",
                               "Trust..Government.Corruption.":"Trust_Goverment_Corruption","Happiness.Rank":"Happiness_Rank"},inplace=True)
year_2017_last
year_2017_last.columns
a=["Happiness_Rank","Happiness_Score","Economy_GDP_per_Capita","Family","Health_Life_Expectancy","Freedom","Generosity","Trust_Goverment_Corruption"]
year_2017_last=year_2017_last[a]
year_2017_last
year_2017_last.corr()
f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(year_2017_last.corr(),annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
year_2017_last.plot(y="Happiness_Score",linewidth=2,kind="line",figsize=(15,15),color="red",label="Happiness",grid=True)
plt.xlabel("Rank")
plt.ylabel("Score")
plt.title("Happiness score by Ranks")
plt.show()
year_2017_last.Freedom.plot(kind = 'line', color = 'g',label = 'Freedom',linewidth=2,alpha = 0.9,grid = True,linestyle = ':',figsize=(15,15))
year_2017_last.Trust_Goverment_Corruption.plot(color = 'r',label = 'Trust_Goverment_Corruption',linewidth=2, alpha = 0.5,grid = True,linestyle = '-',figsize=(15,15))
plt.legend(loc='upper right')     
plt.xlabel('x axis')    
plt.ylabel('y axis')
plt.title('Line Plot')         
plt.show()
year_2017_last.Family.plot(kind="hist",figsize=(15,15),bins=20,grid=True)

plt.show()
year_2017_last.plot(kind="scatter",x="Economy_GDP_per_Capita",y="Health_Life_Expectancy",grid=True,color="blue",figsize=(15,15))
plt.xlabel("Economy")
plt.ylabel("Health")
plt.title("Correlation Health to Economy")
plt.show()
year_2017_last.plot(kind="bar",y="Generosity",color="g",figsize=(32,18))
plt.xlabel("Rank")
plt.ylabel("Generosity")
plt.title("Generosity")
plt.show()
year_2017_last[["Happiness_Score","Freedom"]].plot(subplots=True,figsize=(32,18),grid=True)
plt.show()
trust_mean=sum(year_2017_last.Trust_Goverment_Corruption)/len(year_2017_last.Trust_Goverment_Corruption)
print("trust mean: ",trust_mean)
year_2017_last["trust_level"]=["high"if i >trust_mean else "low" for i in year_2017_last.Trust_Goverment_Corruption]
data1=year_2017_last.trust_level
data2=year_2017_last.Trust_Goverment_Corruption
conc_data=pd.concat([data1,data2],axis=1)
conc_data.head(20)
year_2017_last.columns
year_2017_last[(year_2017_last["Happiness_Score"]>6) & (year_2017_last["trust_level"]=="low")]
year_2017_last["Happiness_Rank"]=year_2017_last["Happiness_Rank"].astype("int")
year_2017_last.dtypes
year_2017_last.head()
year_2017_last.describe()
year_2017_last[["Generosity"]].boxplot(figsize=(16,9))
plt.show()
b=year_2017_last["Generosity"]>0.6
year_2017_last[b]