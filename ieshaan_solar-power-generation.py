# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

data
data=data.drop(columns=['PLANT_ID'],axis=1)

data
data.shape
data.isna().sum()
data.describe()
data.info
data.corr()
sns.heatmap(data.corr(),annot=True,linewidth=0.30,cmap='RdYlGn')
data.hist(figsize=(10,10),edgecolor="k")

plt.tight_layout()

plt.show()
data1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

data1.head()
data1=data1.drop(columns=['PLANT_ID'],axis=1)
data1.describe()
data1.isna().sum()
data1.corr()
sns.heatmap(data1.corr(),annot=True,linewidth=0.30,cmap='RdYlGn')
data1.hist(figsize=(10,10),edgecolor="k")

plt.tight_layout()

plt.show()
df = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')

df
df=df.drop(columns=['PLANT_ID'],axis=1)

df
df.isna().sum()
df.describe()
df.info()
df.corr()
sns.heatmap(df.corr(),annot=True,linewidth=0.30,cmap='RdYlGn')
df.hist(figsize=(10,10),edgecolor="k")

plt.tight_layout()

plt.show()
df1=pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df1
df1=df1.drop(columns=['PLANT_ID'],axis=1)
df1.isna().sum()
df1.describe()
df1.info()
sns.heatmap(df1.corr(),annot=True,linewidth=0.30,cmap='RdYlGn')
df1.hist(figsize=(10,10),edgecolor="k")

plt.tight_layout()

plt.show()