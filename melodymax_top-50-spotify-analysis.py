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

import seaborn as sns

import matplotlib.pyplot as plt
# df = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding='latin_1') 

df = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', index_col=0, encoding='latin_1') 
df.head()
df.columns
df.info()
df.describe()
df.sort_values(by='Popularity',ascending=False)[:5]
df['Artist.Name'].value_counts().plot.bar(figsize=(12,5), color='r')

plt.title('Number of Songs by Artist', size=15)
df['Genre'].value_counts().plot.bar(figsize=(12,5), color='green')

plt.title('Number of Songs by Genre', size=15)
sns.distplot(df['Popularity']) 
plt.style.use('bmh')

df.plot.box(subplots=True, layout=(4,3),figsize=(25,15),color='r',fontsize=15,legend = True)
plt.figure(figsize = (20,7))

sns.heatmap(df.corr(),annot=True)