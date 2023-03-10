import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
d5 = pd.read_csv('../input/2015.csv')
d6 = pd.read_csv('../input/2016.csv')
d7 = pd.read_csv('../input/2017.csv')
d5.isnull().any()
d6.isnull().any()
d7.isnull().any()
d5.sort_values(by=['Happiness Rank']).head()
d6.sort_values(by=['Happiness Rank']).head()
d7.sort_values(by=['Happiness.Rank']).head()
corr5 = d5.corr()
corr5
plt.subplots(figsize=(18, 18))
sns.heatmap(corr5, vmax=.9,annot=True,linewidths=.5)
corr6 = d6.corr()
corr6
plt.subplots(figsize=(18, 18))
sns.heatmap(corr6,vmax=.9,annot=True,linewidths=.5)
corr7=d7.corr()
corr7
plt.subplots(figsize=(18,18))
sns.heatmap(corr7,annot=True,linewidths=.5)
ax1 = d6.plot(kind='scatter', x='Freedom', y="Happiness Rank",alpha = 0.5,color = 'red',figsize=(12,9),subplots = (3,1,1))
ax2 = d5.plot(kind='scatter', x='Freedom', y="Happiness Rank",alpha = 0.5,color = 'black',figsize=(12,9),subplots = (3,1,1),ax=ax1)
ax3 = d7.plot(kind='scatter', x='Freedom', y="Happiness.Rank",alpha = 0.5,color = 'green',figsize=(12,9),subplots = (3,1,1),ax=ax1)
print(ax1==ax2==ax3)
ax1 = d5.plot(kind='scatter', x="Economy (GDP per Capita)", y='Happiness Score',alpha = 0.5,color = 'red',figsize=(12,9),subplots = (3,1,1))
ax2 = d6.plot(kind='scatter', x="Economy (GDP per Capita)", y='Happiness Score',alpha = 0.5,color = 'black',figsize=(12,9),subplots = (3,1,1),ax=ax1)
ax3 = d7.plot(kind='scatter', x="Economy..GDP.per.Capita.", y='Happiness.Score',alpha = 0.5,color = 'green',figsize=(12,9),subplots = (3,1,1),ax=ax1)
print(ax1==ax2==ax3)
ax1 = d5.plot(kind='scatter', x='Family', y="Happiness Rank",alpha = 0.5,color = 'red',figsize=(12,9),subplots = (3,1,1))
ax2 = d6.plot(kind='scatter', x='Family', y="Happiness Rank",alpha = 0.5,color = 'black',figsize=(12,9),subplots = (3,1,1),ax=ax1)
ax3 = d7.plot(kind='scatter', x='Family', y="Happiness.Rank",alpha = 0.5,color = 'green',figsize=(12,9),subplots = (3,1,1),ax=ax1)
print(ax1==ax2==ax3)
ax1 = d5.plot(kind='scatter', x="Trust (Government Corruption)", y="Happiness Rank",alpha = 0.5,color = 'red',figsize=(12,9),subplots = (3,1,1))
ax2 = d6.plot(kind='scatter', x="Trust (Government Corruption)", y="Happiness Rank",alpha = 0.5,color = 'black',figsize=(12,9),subplots = (3,1,1),ax=ax1)
ax3 = d7.plot(kind='scatter', x="Trust..Government.Corruption.", y="Happiness.Rank",alpha = 0.5,color = 'green',figsize=(12,9),subplots = (3,1,1),ax=ax1)
print(ax1==ax2==ax3)
