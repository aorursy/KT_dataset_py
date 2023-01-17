# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_excel("../input/airquality/datasets_100189_236788_AirQuality.xlsx")
data.head(10)
data.info()
data["Pollutants"].unique()
data["State"].unique()
data['city'].unique()
des=pd.DataFrame(data['Avg'])
len(des)
a=list(range(1,825))

fig,axes=plt.subplots(figsize=(20,10))
axes=plt.plot(a,des)
plt.xlabel("s.no")
plt.ylabel('Avg pollution')
plt.figure(figsize=(17,7), dpi = 100)
sns.countplot(x='State',data=data)
plt.xlabel('State')
plt.tight_layout()

by_state = data.groupby('Pollutants')
new_data=by_state.mean()
new_data





plt.figure(figsize=(20,7), dpi = 100)
plt.plot(by_state.mean(), marker='o',)
plt.tight_layout()
plt.legend(['Avg', 'Max', 'Min'])
plt.xlabel('State')
plt.title('Avg Pollution by State')
by_state.max()
plt.figure(figsize=(17,7), dpi = 100)
#by_state.mean().plot()
plt.plot(by_state.max()['Avg'])
plt.plot(by_state.max()['Max'])
plt.plot(by_state.max()['Min'])
plt.tight_layout()
plt.legend(['Avg', 'Max', 'Min'])
plt.xlabel('State')
plt.title('Max Pollution by State')
by_state.max()
plt.figure(figsize=(17,7), dpi = 100)
#by_state.mean().plot()
plt.plot(by_state.min()['Avg'])
plt.plot(by_state.min()['Max'])
plt.plot(by_state.min()['Min'])
plt.tight_layout()
plt.legend(['Avg', 'Max', 'Min'])
plt.xlabel('State')
plt.title('Max Pollution by State')
state = list(data['State'].unique())
fig, axes = plt.subplots(nrows=5,ncols=4,figsize=(17,20))
i = 0
for st in state:
    airQualityState = data[data['State'] == st]
    plot = sns.countplot(x='Pollutants', data = data, ax=axes.flatten()[i])
    plot.set_title(st)
    plt.tight_layout()
    i = i + 1
list(data['Pollutants'].unique())
pollutant = list(data['Pollutants'].unique())
for poll in pollutant:
    plt.figure(figsize=(17,7), dpi = 100)
    sns.countplot(data[data['Pollutants']==poll]['State'], data = data)
    plt.tight_layout()
    plt.title(poll)
max_value = list()
avg_value = list()
min_value = list()
pollutant = list(data['Pollutants'].unique())
for poll in pollutant:
    max_value.append(data[data['Pollutants'] == poll]['Max'].max())
    avg_value.append(data[data['Pollutants'] == poll]['Avg'].mean())
    min_value.append(data[data['Pollutants'] == poll]['Min'].min())
fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(17,7))
axes[0].set_ylabel('Max Value')
axes[0].bar(pollutant, max_value)
axes[0].plot(pollutant, max_value)
axes[1].set_ylabel('Avg Value')
axes[1].bar(pollutant, avg_value)
axes[1].plot(pollutant, avg_value,marker='o',linewidth=3)
axes[2].set_ylabel('Min Value')
axes[2].bar(pollutant, min_value)
axes[2].plot(pollutant, min_value)
