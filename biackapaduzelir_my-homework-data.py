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
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head()
df.info()
df.corr()
f,ax = plt.subplots(figsize=(20,25))
sns.heatmap(df.corr(), annot=True, linewidths= .5, fmt=".1f", ax=ax)
plt.show()
df.columns
df.chol.plot(kind='line', color='purple', label='chol', linewidth=1, alpha=1, grid=True, linestyle=':')
df.trestbps.plot(color='red', label= 'trestbps', linewidth=1, alpha=1, grid=True, linestyle='--')
plt.legend(loc= 'best')
plt.xlabel('x axis') #chol
plt.ylabel('y axis') #trestbps
plt.show()
df.plot(kind='scatter', x='thalach', y='restecg', alpha= 0.5, color='blue')
plt.xlabel('thalach')
plt.ylabel('restecg')
df.age.plot(kind='hist', bins=50, figsize=(15,15))
plt.show()
plt.pie(df['sex'].value_counts(),labels=df['sex'].unique())
#plt.pie(df['age'].value_counts(),labels=df['age'].unique())
plt.show()
df[np.logical_and(df['chol'] > 300, df['sex'] == 0)]
df['speed-thalach'] = ['High' if i>150 else 'Low' if i>130 else 'Regular'  for i in df.thalach]
df.loc[:15, ['speed-thalach','thalach']]