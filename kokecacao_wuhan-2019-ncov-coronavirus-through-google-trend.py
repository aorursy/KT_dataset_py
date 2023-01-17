from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
print(os.listdir('../input/2019-ncov'))
csv = pd.read_csv("../input/2019-ncov/2019-nCoV.csv")

csv.head()
csv['Date'] = pd.to_datetime(csv.Date)

ax = sns.lineplot(data=csv, x='Date',y='肺炎')

ax.set_title('Googling trend of Pneumonia topic')

ax.set_ylabel('Scaled number of Googling from China')
csv['Date'] = pd.to_datetime(csv.Date)

ax = sns.lineplot(data=csv, x='Date',y='流行性感冒')

ax.set_title('Googling trend of Pneumonia topic')

ax.set_ylabel('Scaled number of Googling from China')
csv['Date'] = pd.to_datetime(csv.Date)

ax = sns.lineplot(data=csv, x='Date',y='病毒')

ax.set_title('Googling trend of Virus topic')

ax.set_ylabel('Scaled number of Googling from China')
csv['Date'] = pd.to_datetime(csv.Date)

ax = sns.lineplot(data=csv, x='Date',y='武汉市')

ax.set_title('Googling trend of Wuhan topic')

ax.set_ylabel('Scaled number of Googling from China')
csv['Date'] = pd.to_datetime(csv.Date)

ax = sns.lineplot(data=csv, x='Date',y='冠状病毒')

ax.set_title('Googling trend of Coronavirus topic')

ax.set_ylabel('Scaled number of Googling from China')
plt.plot( 'Date', '流行性感冒', data=csv, marker='', markerfacecolor='yellow', color='skyblue', linewidth=1, label="Flu")

plt.plot( 'Date', '病毒', data=csv, marker='', markerfacecolor='blue', color='skyblue', linewidth=1, label="Virus")

plt.plot( 'Date', '武汉市', data=csv, marker='', color='olive', linewidth=1, label="WuHan")

plt.plot( 'Date', '冠状病毒', data=csv, marker='', color='green', linewidth=1, linestyle='dashed', label="Coronavirus")

plt.legend()
