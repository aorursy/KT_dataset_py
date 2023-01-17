# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
space=pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')
space.head()
space.shape
space.drop(['Unnamed: 0','Unnamed: 0.1'], inplace=True, axis=1)
space.head()
space.info()
space['Country']=space['Location'].apply(lambda x: str(x).split(', ')[-1])
space.head()
space.isna().sum()
space['Year']=space['Datum'].apply(lambda x: str(x).split(', ')[-1])

space['Month']=space['Datum'].apply(lambda x: str(x).split(', ')[0])

space['Datum_year']=space['Year'].apply(lambda x: str(x).split(' ')[0])

space['Datum_month']=space['Month'].apply(lambda x: str(x).split(' ')[-2])

space.head()
plt.figure(figsize=(20,6))

space['Company Name'].value_counts().plot(kind='bar')
space['Status Rocket'].value_counts().plot(kind='bar')
space['Status Mission'].value_counts().plot(kind='bar')
space['Country'].value_counts().plot(kind='bar')
plt.figure(figsize=(20,6))

sns.countplot(space['Datum_year'])

plt.xticks(rotation=90)
space['Datum_month'].value_counts().plot(kind='bar')
space[space['Company Name']=='SpaceX']['Status Rocket'].value_counts()
space[space['Company Name']=='SpaceX']['Status Rocket'].value_counts().plot(kind='bar')
space[space['Company Name']=='SpaceX']['Status Mission'].value_counts()
space[space['Company Name']=='SpaceX']['Status Mission'].value_counts().plot(kind='bar')
space[space['Company Name']=='SpaceX']['Datum_year'].value_counts()
space[space['Company Name']=='SpaceX']['Datum_year'].value_counts().plot(kind='bar')
space[space['Company Name']=='ISRO']['Status Rocket'].value_counts()
space[space['Company Name']=='ISRO']['Status Rocket'].value_counts().plot(kind='bar')
space[space['Company Name']=='ISRO']['Status Mission'].value_counts()
space[space['Company Name']=='ISRO']['Status Mission'].value_counts().plot(kind='bar')
space[space['Company Name']=='ISRO']['Datum_year'].value_counts()
space[space['Company Name']=='ISRO']['Datum_year'].value_counts().plot(kind='bar')