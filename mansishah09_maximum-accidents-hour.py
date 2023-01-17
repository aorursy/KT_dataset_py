# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import datetime 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_Dec19.csv")
df.shape
df.isnull().sum()
df['Dates']=pd.to_datetime(df['Start_Time']).dt.date

print(df['Dates'])
df['Month']=pd.DatetimeIndex(df['Dates']).month

df['Month']
df.groupby(df['Month'])['ID'].count().sort_values(ascending=False)
sns.countplot(x='Month',data=df)