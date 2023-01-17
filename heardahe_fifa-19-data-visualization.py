# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import pi



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/data.csv")

df.info()
ColumnNames = list(df.columns.values)

ColumnNames
C_Data = pd.concat([df[['ID','Name','Preferred Foot','Position','Age','Overall','Value']],df[ColumnNames[54:88]]],axis=1)

C_Data.isnull().sum()
C_Data[C_Data.Position.isnull()]
C_Data = C_Data.dropna()
def CleanSalary(param1):

    param1 = param1.replace('€','')

    if 'K' in param1:

        param1 = float(param1.replace('K',''))*1000

    elif 'M' in param1:

        param1 = float(param1.replace('M',''))*1000000

    return float(param1)



C_Data['Value'] = C_Data['Value'].apply(lambda x: CleanSalary(x))
import seaborn as sns

import matplotlib.pyplot as plt



from mpl_toolkits.mplot3d import Axes3D 

%matplotlib inline
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(C_Data['Age'],C_Data['Overall'],C_Data['Value'],c='b',marker='.')

ax.set_xlabel('Age', fontsize=12)

ax.set_ylabel('Player Level',fontsize=12,rotation=90)

ax.set_zlabel('Value',fontsize=12,rotation=90)

ax.view_init(15,45)
C_Data = pd.concat([df[['Position','Overall']],df[ColumnNames[54:88]]],axis=1)

HeatmapData = C_Data.groupby('Position').mean()

sns.heatmap(HeatmapData,cmap='Blues',xticklabels = True,yticklabels = True)
labels = np.array(HeatmapData.columns.values)

N = len(labels)



Position = 'CB'

stats=HeatmapData.loc[Position,labels]



angles = [n / float(N) * 2 * pi for n in range(N)]



stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))





fig=plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=1)

ax.fill(angles, stats, alpha=0.9)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title(Position)

ax.grid(True)
labels = np.array(HeatmapData.columns.values)

N = len(labels)



Position = 'CAM'

stats=HeatmapData.loc[Position,labels]



angles = [n / float(N) * 2 * pi for n in range(N)]



stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))





fig=plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=1)

ax.fill(angles, stats, alpha=0.9)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title(Position)

ax.grid(True)
labels = np.array(HeatmapData.columns.values)

N = len(labels)



Position = 'ST'

stats=HeatmapData.loc[Position,labels]



angles = [n / float(N) * 2 * pi for n in range(N)]



stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))





fig=plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=1)

ax.fill(angles, stats, alpha=0.9)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title(Position)

ax.grid(True)
labels = np.array(HeatmapData.columns.values)

N = len(labels)



Position = 'GK'

stats=HeatmapData.loc[Position,labels]



angles = [n / float(N) * 2 * pi for n in range(N)]



stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))





fig=plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=1)

ax.fill(angles, stats, alpha=0.9)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title(Position)

ax.grid(True)