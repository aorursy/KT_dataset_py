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
#import data
all_data = pd.read_csv("../input/athlete_events.csv")
data=all_data[(all_data.Sport=="Basketball") & (all_data.Year>=1990)]



data.info()
#correlation
f,ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head()
data.columns
data.describe()
#Medal data
gold_medal=data[data.Medal=="Gold"]
silver_medal=data[data.Medal=="Silver"]
bronze_medal=data[data.Medal=="Bronze"]

#Reset index
gold_medal = gold_medal.reset_index()
silver_medal = silver_medal.reset_index()
bronze_medal = bronze_medal.reset_index()





#Reset index
data = data.reset_index()
#Age Plot
data.Age.plot(kind='hist',bins=40,color = 'g',figsize=(15,5))
plt.show()
#Medal Plots for Height
gold_medal.Age.plot(kind='line',label="Gold",alpha = 0.5,color = 'gold',figsize=(15,5))
silver_medal.Age.plot(kind='line',label="Silver",alpha = 0.5,color = 'silver')
bronze_medal.Age.plot(kind='line',label="Bronze",alpha = 0.5,color = 'green')
plt.legend()
plt.show()

#Weight-Height correlation plot
data.plot(kind="scatter",x="Height",y="Weight")
plt.show()

#Winner Countries
countries=[]
for i in gold_medal.Team:
    if i not in countries:
        countries+=[i]
print(countries)
#Loop for 'how many player have gold medal to countries?'
us_gold=[]
for i in gold_medal.Team:
    if i=="United States":
        us_gold+=[i]
uf_gold=[]
for i in gold_medal.Team:
    if i=="Unified Team":
        uf_gold+=[i]
argentina_gold=[]
for i in gold_medal.Team:
    if i=="Argentina":
        argentina_gold+=[i]
medal_count=[len(us_gold),len(uf_gold),len(argentina_gold)]




#Players medal count's plot
plt.bar(countries,medal_count)
plt.show()