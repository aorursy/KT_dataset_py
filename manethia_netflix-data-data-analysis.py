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
df=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
df.head()
df.info()
df.isnull().sum()
df.drop(['director', 'cast'], axis=1,inplace=True)
df.head()
df.info()
df.isnull().sum()
df.duplicated()
df.duplicated().sum()
df.head()
df['type'].count
import seaborn as sns

import matplotlib.pyplot as plt
sns.countplot(x='type',data=df)

plt.title("Total no of Movies and Tv Shows");
plt.figure(figsize=(11,9))

sns.countplot(x='rating',data=df)

plt.title("Ratings");
plt.figure(figsize=(12,12))

sns.countplot(x='type',hue='rating',data=df)

plt.title("Comparing type and rating of tv shows and movies");
