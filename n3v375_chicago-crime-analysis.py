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

from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('../input/chicago-crime-from-01jan2001-to-22jul2020/Crimes_-_2001_to_Present.csv', low_memory=False)
df.head()
df.dropna(axis=1, inplace=True)

df
df.info()
df.describe()
df['Date'].value_counts()[:10]
df['Block'].value_counts()[:10]
df['Primary Type'].value_counts()[:10]
df['Description'].value_counts()[:10]
df['Arrest'].value_counts()
df['Domestic'].value_counts()
df['Year'].value_counts()
sns.clustermap(df.corr(), annot=True, cmap='plasma')
le = LabelEncoder()

df['block_cat'] = le.fit_transform(df['Block'])

df['type_cat'] = le.fit_transform(df['Primary Type'])

df['desc_cat'] = le.fit_transform(df['Description'])

df['arrest_cat'] = le.fit_transform(df['Arrest'])

df['dom_cat'] = le.fit_transform(df['Domestic'])

df['fbi_cat'] = le.fit_transform(df['FBI Code'])

plt.figure(figsize=(18,6))

sns.heatmap(df.corr(), annot=True, cmap='plasma')
plt.figure(figsize=(18,6))

sns.countplot(df['Year'], hue=df['Arrest'])

plt.title('Arrests made over the years')