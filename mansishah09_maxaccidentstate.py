# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

df=pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
#No of attributes and no of rows(datasets)

df.shape
df.head(10)
df.isnull().sum()

#As state does not contain null values it is fine to use it
#Group according to state and id to find no of accidents in each state

df=df.groupby('State')['ID'].count().reset_index()

print(df)

df.shape
#sort to find the state with maximum no of accidents

df=df.sort_values(by="ID",ascending=False)

df

#plot the ratio of state to accidents in descending order in bar graph format

fig=plt.figure(figsize=(35,10))

plt.bar(x=df['State'],height=df['ID'],width=0.6)

plt.xticks(df['State'])