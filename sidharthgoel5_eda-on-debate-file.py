# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Visualization of the data

import seaborn as sns # Visualization of the data

%matplotlib inline 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_debate = pd.read_csv('..//input//democratic-debate-transcripts-2020//debate_transcripts.csv',encoding='cp1252')
df_debate.head()
df_debate.shape
df_debate.info()
df_debate['speaker'].value_counts()
df_debate.isnull().sum()
df_debate[df_debate.isna().any(axis=1)]
df_debate.dropna(axis=0,inplace=True)
df_debate.shape
df_debate_above_1 = df_debate[df_debate['speaker'] > '1']
df_debate_above_1['speaker'].unique()
df_debate_above_1 = df_debate.groupby(['speaker'])['speaking_time_seconds'].sum()

df_debate_above_1.sort_values(inplace=True)
plt.figure(figsize=(20,25))

df_debate_above_1.plot.barh()

#df_debate_above_1.set_ytickslable(y_labels)
ax = df_debate['debate_name'].value_counts().plot(kind='barh')

ax.set_alpha(0.8)

ax.set_title("Bar Graph", fontsize=22)

ax.set_ylabel("Debate Topic", fontsize=15);

plt.show()
df_debate.groupby(['debate_name'])['speaking_time_seconds'].mean()