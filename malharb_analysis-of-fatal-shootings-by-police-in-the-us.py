# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
plt.figure(figsize=(10,7),dpi=200)
sns.countplot(x='year',data=df,hue='month')
plt.legend(bbox_to_anchor=(1,1))
plt.ylabel('killings')
plt.title('FATAL SHOOTINGS EACH YEAR BY MONTH')
plt.figure(figsize=(10,7),dpi=200)
sns.distplot(df.dropna(subset=['age'])['age'])
plt.title('AGE DISTRIBUTION OF VICTIMS')
plt.figure(figsize=(10,7),dpi=200)
sns.countplot(x='gender',data=df)
plt.ylabel('killings')
plt.title('DEATHS BY GENDER')
deathsbyrace = [list(df['race'].value_counts()),['White','Black','Hispanic','Asian','Native American','Other']]
plt.figure(figsize=(10,7),dpi=200)
plt.pie(x=deathsbyrace[0],labels=deathsbyrace[1],wedgeprops={'linewidth':2,'edgecolor':'black'})
plt.title('DEATHS BY RACE')
plt.figure(figsize=(10,7),dpi=200)
sns.countplot(x='flee',data=df,palette='coolwarm')
plt.ylabel('killings')
plt.xlabel('')
plt.title('FLEEING VICTIMS')
plt.figure(figsize=(10,7),dpi=200)
sns.countplot(x='state',data=df)
plt.xticks(rotation=90)
plt.ylabel('killings')
plt.title('DEATHS BY STATE')
plt.tight_layout()
plt.figure(figsize=(10,7),dpi=200)
data=pd.DataFrame(df.groupby('state').mean()['age'])
sns.barplot(x=data.index,y='age',data=data,palette='icefire')
plt.xticks(rotation=90)

plt.title('MEAN AGE OF VICTIMS BY STATE')
plt.tight_layout()

plt.savefig('mAge_by_state.png')
