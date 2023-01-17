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
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/human-resources-data-set/HRDataset_v13.csv')
df.head()
df.info()
df.describe()
# Visualize and answer questions given in the link https://www.kaggle.com/rhuebner/human-resources-data-set
df['RecruitmentSource'].unique()
df['RecruitmentSource'].isna().sum()
df1 = df[['RecruitmentSource','Position','PerformanceScore']]
df1.dropna(inplace=True)
df1.isna().sum()
df1.PerformanceScore.unique()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['PerformanceScore'] = le.fit_transform(df1['PerformanceScore'])
df1.head()
plt.figure(figsize=(18,8))
plt.xticks(rotation=45)
sns.countplot(x='Position',hue='PerformanceScore',data=df1)
df1 = df['Position','PerformanceScore']
# Question2. What is the overall diversity profile of the organization?
plt.figure(figsize=(22,8))
plt.xticks(rotation=45)
sns.countplot(x='Position',data=df1)

# Question 3. What are our best recruiting sources if we want to ensure a diverse organization?
plt.figure(figsize=(18,8))
plt.xticks(rotation=45)
sns.countplot(x='RecruitmentSource',hue='PerformanceScore',data=df1)
# Question4. Can we predict who is going to terminate and who isn't? 
# What level of accuracy can we achieve on this?

df.columns

df4 = d