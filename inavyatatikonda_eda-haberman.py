# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/habermans-survival-data-set/haberman.csv')

df.head()
df=df.rename(columns={"30": "Age", "64": "Op_Year","1":"axil_nodes","1.1":"Surv_status"})
df.head()
df.info()

df.shape
#Check how many of them have survived

df['Surv_status'].value_counts()
#Check how many of them have survived

df['Surv_status'].value_counts().plot.pie(explode=[0.01,0.01],autopct="%.1f%%")
df['Age'].value_counts()
df['Age'].max()

df['Age'].min()
#Range

df['Age'].max()-df['Age'].min()
#Frequency Distribution Tables.

df1 = pd.Series(df['Age']).value_counts().sort_index().reset_index().reset_index(drop=True)

df1.columns = ['Age','Surv_status']

print(df1)

#Bar Charts.



df.plot(kind="scatter",x="Age",y="axil_nodes")
#Bar Charts.



df.plot(kind="bar",y="Age",x="Surv_status",color="blue")
sns.set(style="whitegrid")

sns.barplot(y="Age",x="Surv_status",hue="Age",data=df)
sns.set(style="whitegrid")

sns.barplot(y="Age",x="Surv_status",data=df)
sns.distplot(df['Age'],rug=True)

#RugPlot is used to display the distribution of the data
sns.kdeplot(df['Age'],shade="True")