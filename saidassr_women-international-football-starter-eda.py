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
#reading CSV file as dataframe

df = pd.read_csv('/kaggle/input/womens-international-football-results/results.csv', index_col = 0)

#view data

df.head()
df.isnull().sum()
df_copy = df

df_copy['date'] = df_copy.index

df_copy['year'] = pd.DatetimeIndex(df_copy['date']).year

df_copy = df_copy[['year','tournament']]

#df_copy.head()
import matplotlib.pyplot as plt



df_tournament= df_copy.groupby(['year','tournament']).count().groupby(level='year').size().rename('count')

plt.figure(figsize=(15,6))

df_tournament.plot.bar()

plt.ylabel('No of Tournaments')

plt.title('No. of tournaments per year')

plt.show()
df_tournament[df_tournament==np.max(df_tournament)].index[0]
np.mean(df_tournament[df_tournament.index>=2010])