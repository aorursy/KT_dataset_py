# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/denver-crime-data/crime.csv')

df2 = pd.read_csv('../input/denver-crime-data/offense_codes.csv')

df
df.groupby('NEIGHBORHOOD_ID')['INCIDENT_ID'].count()
#indian-creek has least number of incidents

df.groupby('NEIGHBORHOOD_ID')['INCIDENT_ID'].count().reset_index(name='count').sort_values(['count']).head()
#five-points have the highest number of incidents

df.groupby('NEIGHBORHOOD_ID')['INCIDENT_ID'].count().reset_index(name='count').sort_values(['count'], ascending=False).head()



#or df.groupby('NEIGHBORHOOD_ID')['INCIDENT_ID'].count().reset_index(name='count').sort_values(['count']).tail()
df['OFFENSE_CATEGORY_ID'].unique()
#clearly, traffic violations have been the highest

ax = df['OFFENSE_CATEGORY_ID'].value_counts().plot(kind='bar',

                                    figsize=(14,8),

                                    title="Bar Plot")

ax.set_xlabel("OFFENSE_CATEGORY_ID")

ax.set_ylabel("Frequency")