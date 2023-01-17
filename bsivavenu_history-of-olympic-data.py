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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
events = pd.read_csv('../input/athlete_events.csv')
events.shape
events.head()
region = pd.read_csv('../input/noc_regions.csv')
region.shape
region.head()
events.isnull().sum()
events.Sex.value_counts()
events.Year.value_counts()
plt.figure(figsize=(20,5))
sns.pointplot('Year',y = events.ID.index.unique(),hue = 'Sex',data=events,dodge= True)
plt.show()
plt.figure(figsize=(20,5))
events.Year.value_counts().plot(kind = 'bar')