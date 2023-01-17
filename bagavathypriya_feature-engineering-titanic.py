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
df=pd.read_csv("../input/titanicdataset-traincsv/train.csv")
import pandas_profiling
pandas_profiling.ProfileReport(df)
df.head()
df.isnull().sum()
df[df['Embarked'].isnull()]
df['Cabin_null']=np.where(df['Cabin'].isnull(),1,0)

df['Cabin_null'].mean()
df.groupby(['Survived'])['Cabin_null'].mean()
df=pd.read_csv("../input/titanicdataset-traincsv/train.csv",usecols=['Age','Fare','Survived'])
df.isnull().mean()
def impute_median(df,variable,median):
    df[variable+'med']=df[variable].fillna(median)
med=df.Age.median()
med
impute_median(df,'Age',med)
df[df['Age'].isnull()]
print(df['Age'].std())
print(df['Agemed'].std())
fig = plt.figure() 
ax = fig.add_subplot(111) 
df['Age'].plot(kind='kde', ax=ax) 
df.Agemed.plot(kind='kde', ax=ax, color='red') 
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')
