# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/StudentsPerformance.csv')
sns.heatmap(df.isna())
df.info()
df.corr()
df.groupby('parental level of education')[['math score', 'reading score','writing score']].mean().plot(kind='bar')
plt.margins(x=0, y=-0.0)   
df.groupby('race/ethnicity')[['math score', 'reading score','writing score']].mean().plot(kind='bar')
plt.margins(x=0, y=-0.0)   
df.groupby('test preparation course')[['math score', 'reading score','writing score']].mean().plot(kind='bar')
plt.margins(x=0, y=-0.0)   
dfvl = df['gender'].value_counts().plot(kind='bar')
plt.grid(True)
df.groupby('gender')[['math score', 'reading score','writing score']].mean().plot(kind='bar')
plt.margins(x=0, y=-0.0)   
plt.grid(True)
sms = df[df['gender']=='male'].groupby('gender')['math score'].sum()
sfs =  df[df['gender']=='female'].groupby('gender')['math score'].sum()

print(sms[0] * 100 / sfs[0] / 100)