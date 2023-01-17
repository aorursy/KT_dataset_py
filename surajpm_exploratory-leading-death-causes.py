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
Data = pd.read_csv('../input/selected-trend-table-from-health-united-states-2011.-leading-causes-of-death-and-numbers-of-deaths-by-sex-race-and-hispanic-origin-united-states-1980-and-2009.csv')                   
Data.info()
Data['Year'].value_counts()
Data['Group'].value_counts()
Data['Cause of death'].value_counts()
Data.groupby(['Year','Group']).sum()['Deaths']#.plot('bar')
df = pd.DataFrame(Data.groupby(['Year','Cause of death']).sum()['Deaths'].sort_values(ascending=False).head(20)).reset_index()#.plot('bar')

df
import matplotlib.pyplot as plt

import seaborn as sns
sns.barplot(y='Cause of death', x='Deaths', data=df, hue='Year')

plt.title('Leading causes of death')

#plt.xticks(rotation = 90