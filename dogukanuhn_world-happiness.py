

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#reading csv for examination
year2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")
year2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")
year2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")
year2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")
year2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
year2019.info()
#data types in object
year2019.dtypes
#column names
year2019.columns
#correlation of data
year2019.corr()
#top 5 
year2019.head()

#last 5 
year2019.tail()
#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(year2019.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
year2019.plot(kind='scatter', x='Freedom to make life choices', y='Healthy life expectancy',alpha = 0.5,color = 'dimgray')
plt.xlabel('Freedom to make life choices')              
plt.ylabel('Healthy life expectancy')
plt.title('Healthy life expectancy And Freedom to make life choices')
plt.show()
#regions of world
regions = year2019["Country or region"]
print("Regions Of World:")
for x in regions.unique().tolist():
    print(x)
# Happiness Score bigger than 5, Happiness Rank bigger than 4 and Happiness Rank lower than 10 
year2019[(year2019['Freedom to make life choices']>0) & (year2019['Healthy life expectancy']>0) & (year2019['Healthy life expectancy']<0.5)]

year2019_region =  year2019.loc[:,"Country or region"] 
print(year2019_region)
print(year2019["Freedom to make life choices"].value_counts(dropna=False))
year2019.describe()
year2019.boxplot(column="Freedom to make life choices", by="Generosity")
data_new = year2019.head()
melted = pd.melt(frame=data_new,id_vars= "Country or region", value_vars= ["Freedom to make life choices" , "Generosity"])
melted
data1 = year2019.head()
data2 = year2019.tail()
conc = pd.concat([data1,data2],axis=0,ignore_index=True)
conc
data1 = year2019["Social support"].head()
data2 = year2019["Score"].head()
conc = pd.concat([data1,data2],axis=1)
conc