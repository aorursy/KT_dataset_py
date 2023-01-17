# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/world-happiness-report-2019.csv')

df.head()
import seaborn as sns

sns.pairplot(df)
import matplotlib.pyplot as plt

dmor = df[df['Country (region)']=='Morocco']
print(np.array(dmor))
print(dmor['Country (region)'])
sns.barplot(data = dmor,orient='h')
dmaghreb = df[(df['Country (region)']=='Morocco') | (df['Country (region)']=='Tunisia') | 

              (df['Country (region)']=='Algeria') | (df['Country (region)']=='Lybia')]
dmaghreb
sns.barplot(data = dmaghreb,orient='h')
sns.barplot(data = dmaghreb,orient = 'v', x ='Country (region)' ,y = 'Corruption')
sns.barplot(data = dmaghreb,orient = 'v' , x ='Country (region)' ,y = 'Positive affect' , hue = 'Negative affect')