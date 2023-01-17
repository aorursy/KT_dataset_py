# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns#diagrams



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_Dec19.csv")

df.shape
df.head(10)
df.columns.values
df.describe()
df1=df.groupby('State')['ID'].count().reset_index()

df1
figure=plt.subplots(figsize=(10,15))

sns.barplot(y="State",x="ID",data=df1)
corr1=df.corr()

figure=plt.subplots(figsize=(12,12))

sns.heatmap(corr1)
figure=plt.subplots(figsize=(8,10))

sns.countplot(x="Severity",data=df)