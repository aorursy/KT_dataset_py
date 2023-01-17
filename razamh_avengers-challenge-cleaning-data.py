# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
avengers = pd.read_csv("/kaggle/input/avengers/avengers.csv", encoding = "latin-1")
avengers.head()
avengers.columns
avengers.dtypes
avengers['Year'].hist()
avengers["Year"].describe()
avengers = avengers[avengers["Year"]>= 1960]
avengers['Year'].hist()

plt.show()
avengers.head()
def deaths(series):

    death_count = 0

    cols = ["Death1", "Death2", "Death3", "Death4", "Death5"]

    counts = 0

    for i in cols:

        if series[i] == "NO" or pd.isnull(series[i]):

            continue

        else:

            counts+=1

    return counts

    
avengers["Deaths"] = avengers.apply(deaths, axis = 1)
cols = ["Death1", "Death2", "Death3", "Death4", "Death5"]

counts = 0

for i in cols:

    if avengers.iloc[100][i] == "NO" or pd.isnull(avengers.iloc[100][i]):

        continue

    else:

        counts+=1

counts
avengers.iloc[100]
pd.options.display.max_columns = None

avengers.head()
avengers['Years since joining'].values
count = 0

for i, row in avengers[['Years since joining', 'Year']].iterrows():

    if ~np.isnan(row['Year']) or ~np.isnan(row['Years since joining']):

        years_joined = 2015 - int(row['Year'])

        if years_joined == row['Years since joining']:

            count += 1

count
joined_accuracy_count  = int()

correct_joined_years = avengers[avengers['Years since joining'] == (2015 - avengers['Year'])]

joined_accuracy_count = len(correct_joined_years)

joined_accuracy_count