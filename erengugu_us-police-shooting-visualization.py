# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/us-police-shootings/shootings.csv")
df.head()
df.info()
df.isnull().sum()
plt.figure(figsize=(10,6))

sns.heatmap(df.isnull());
list = ["id","name","date"]

df = df.drop(list,axis=1)
df.head()
plt.figure(figsize=(10,6))

sns.countplot(df.manner_of_death);
df.armed.value_counts().sort_values(ascending=False).head().index
df.armed.value_counts().sort_values(ascending=False).head().values
plt.figure(figsize=(10,6))

sns.barplot(x=df.armed.value_counts().sort_values(ascending=False).head().index,y=df.armed.value_counts().sort_values(ascending=False).head().values);
plt.figure(figsize=(10,6))

sns.barplot(x=df.age.value_counts().sort_values(ascending=False).head().index,y=df.age.value_counts().sort_values(ascending=False).head().values);
plt.figure(figsize=(8,6))

sns.countplot(df.gender);
plt.figure(figsize=(8,6))

sns.countplot(df.race);
df["city"].value_counts().sort_values(ascending=False)
df.threat_level.value_counts()
df.head()
df.armed.value_counts()
df.drop("state",axis=1,inplace=True)
df.head()
sns.countplot(df.flee)
sns.countplot(df.body_camera)
df.body_camera.value_counts()
sns.heatmap(df.corr(),square=True,annot=True)