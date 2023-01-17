import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/battles.csv')
data_cd = pd.read_csv('../input/character-deaths.csv')
data_cp = pd.read_csv('../input/character-predictions.csv') 
data.info()
data.corr()
f,ax = plt.subplots(figsize=(25,25))
sns.heatmap(data.corr(), annot = True,ax=ax)
plt.show()
data.head()
data.columns
data.major_death.plot(color = "black", label = "major_death",grid = True, linestyle = ':')
data.major_capture.plot(color = "orange", label = "major_capture",grid = True, linestyle = '-.')
plt.legend()

plt.show()
data.plot(kind ='scatter', x='major_death', y='major_capture', alpha = 0.8, color='blue')
plt.show()
data.plot(kind ='scatter', x='attacker_size', y='defender_size', alpha = 0.6, color='blue')
plt.show()
data.major_death.plot(kind = "hist", color ="r", bins=50, figsize=(10,10), grid = True)
plt.show()
data_cd.columns
data_cd.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data_cd.columns]
data_cd.columns
data_cd.Death_Chapter.plot(kind = "hist", color ="b", bins=40, figsize=(10,10), grid = True)
plt.title('Death_Chapter')
plt.xlabel('Death_Chapter')
plt.ylabel('how many death') 
plt.show()
type(data)
series = data["major_death"]
print(type(series))
data_frame = data[["major_death"]]
print(type(data_frame))
data.head(15)