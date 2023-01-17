# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head(5)
data.info()
set(data.dtypes.tolist())
data.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
data.corr()['DEATH_EVENT'][:-1]
sns.catplot(x='DEATH_EVENT',y='age',data=data)
sns.catplot(x='DEATH_EVENT',y='age',data=data,jitter=False)
sns.catplot(x='DEATH_EVENT',y='age',data=data,kind='swarm')
sns.catplot(x='DEATH_EVENT',y='age',hue='sex',data=data,kind='swarm')
sns.catplot(x='smoking',y='age',hue='sex',data=data,kind='swarm')
sns.catplot(x='DEATH_EVENT',y='platelets',hue='sex',data=data,kind='swarm')
sns.catplot(x='high_blood_pressure',y='platelets',hue='sex',data=data,kind='swarm')
sns.catplot(x='high_blood_pressure',y='platelets',hue='diabetes',data=data,kind='swarm')
sns.catplot(x='high_blood_pressure',y='platelets',hue='smoking',data=data,kind='swarm')
x=sns.catplot(x='anaemia',y='platelets',hue='smoking',data=data,kind='swarm')
y=sns.catplot(x='anaemia',y='platelets',hue='sex',data=data,kind='swarm')
z=sns.catplot(x='anaemia',y='platelets',hue='diabetes',data=data,kind='swarm')
cat=['anaemia','diabetes','smoking','sex']
con=['creatinine_phosphokinase','age','ejection_fraction','platelets','serum_creatinine','serum_sodium']
for x in cat:
  for y in con:
      sns.catplot(x='DEATH_EVENT',y=y,hue=x,kind='box',data=data)
cat=['anaemia','diabetes','smoking','sex']
con=['creatinine_phosphokinase','age','ejection_fraction','platelets','serum_creatinine','serum_sodium']
for x in cat:
  for y in con:
      sns.catplot(x='DEATH_EVENT',y=y,hue=x,kind='boxen',data=data)
cat=['anaemia','diabetes','smoking','sex']
con=['creatinine_phosphokinase','age','ejection_fraction','platelets','serum_creatinine','serum_sodium']
for x in cat:
  for y in con:
      sns.catplot(x='DEATH_EVENT',y=y,hue=x,kind='violin',data=data)
cat=['anaemia','diabetes','smoking','sex']
con=['creatinine_phosphokinase','age','ejection_fraction','platelets','serum_creatinine','serum_sodium']
for y in con:
      plot=sns.catplot(x='DEATH_EVENT',y=y,kind='violin',data=data)
      sns.swarmplot(x='DEATH_EVENT',y=y,data=data,color="k", size=3,ax=plot.ax)

cat=['anaemia','diabetes','smoking','sex']
con=['creatinine_phosphokinase','age','ejection_fraction','platelets','serum_creatinine','serum_sodium']
for x in cat:
  for y in con:
      sns.catplot(x='DEATH_EVENT',y=y,hue=x,kind='bar',data=data)
cat=['anaemia','diabetes','smoking','sex','DEATH_EVENT']
for x in cat:
      sns.catplot(x=x,kind='count',data=data)
cat=['anaemia','diabetes','smoking','sex']
con=['creatinine_phosphokinase','age','ejection_fraction','platelets','serum_creatinine','serum_sodium']
for x in cat:
  for y in con:
      sns.catplot(x='DEATH_EVENT',y=y,hue=x,kind='point',data=data,markers=["^", "o"], linestyles=["-", "--"],)