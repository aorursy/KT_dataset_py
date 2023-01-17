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
dataset=pd.read_csv("/kaggle/input/cholera-dataset/data.csv")

dataset.head()

dataset.isnull().sum()

dataset.shape

dataset.dropna(inplace=True)
dataset.isnull().sum()

dataset.shape
import matplotlib.pyplot as plt

import seaborn as sn

plt.figure(figsize=(14,6))

dataset.groupby('Country')['Cholera case fatality rate'].count().sort_values(ascending=False).head(10).plot.bar(color="red")
plt.figure(figsize=(14,6))

dataset.groupby('Year')['Number of reported cases of cholera'].count().sort_values(ascending=False).head(10).plot.bar(color="blue")
dataset.head(10)
plt.figure(figsize=(14,6))

dataset.groupby('Country')['Number of reported deaths from cholera'].count().sort_values(ascending=False).head(10).plot.bar(color="green")
plt.figure(figsize=(14,6))

dataset.groupby('Year')['Number of reported deaths from cholera'].count().sort_values(ascending=False).head(10).plot.bar(color="green")
plt.figure(figsize=(14,6))

dataset['WHO Region'].value_counts().plot.bar(color="violet")

#African region looks somewhat highly imbalanced
dataset.head()

type_s=['1994']

dataset[dataset.Year.isin(type_s)].groupby('Year')['Number of reported cases of cholera'].count().head(4).plot.bar()
dataset.head()
#Analysis on who region

plt.figure(figsize=(14,6))

dataset.groupby('WHO Region')['Cholera case fatality rate'].count().sort_values(ascending=False).head(5).plot.bar(color="yellow")
#Analysis on who region

plt.figure(figsize=(14,6))

dataset.groupby('WHO Region')['Number of reported cases of cholera'].count().sort_values(ascending=False).head(5).plot.bar(color="brown")