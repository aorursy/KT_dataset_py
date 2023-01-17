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
import matplotlib.pyplot as plt

import seaborn as sns
playstore = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
playstore.head()
playstore.info()
for col in playstore.columns:

    print('{} has {} unique values'.format(col, len(playstore[col].unique())))
playstore.isnull().sum()
playstore.loc[playstore['Reviews'] == '3.0M', 'Reviews'] = 3000000

playstore.loc[playstore['Size'] == 'Varies with device', 'Size'] = '-1M'

playstore.loc[playstore['Size'] == '1,000+', 'Size'] = '-1M'

playstore.loc[:, 'Size'] = playstore.loc[:, 'Size'].apply(lambda x : float(x[:-1]) * 1000 if x[-1] == 'M' else float(x[:-1]))
def drawBarChart(df, col):

    tmp = df.reset_index()

    tmp = tmp.groupby(col).index.count()

    tmp.plot(figsize = (15,7), kind = 'bar', label = 'count')

    plt.legend()
playstore.groupby('Category')['App'].count().plot(kind = 'bar', figsize = (15, 7))

_ = plt.title('Number of app with different category')
sns.kdeplot(playstore.groupby('App').agg({'Rating':'mean'})['Rating'], shade = True)

_ = plt.xlim(0,5)

_ = plt.title('rating distribution for all available app')
playstore.loc[playstore['Reviews'] == '3.0M', 'Reviews'] = 3000000

playstore['Reviews'] = playstore['Reviews'].astype('int64')

tmp = playstore.groupby('Category').agg({'Reviews' : 'sum'})

tmp.plot(kind = 'bar', figsize = (15,7))

_ = plt.title('number of user reviws for cagetory')
playstore.groupby('Content Rating')['App'].count().plot(figsize = (15, 7), kind = 'bar')

_ = plt.title('Number of app with different Content ratings')
playstore.groupby(['Installs'])['App'].count().plot(kind = 'bar', figsize = (15,7))

_ = plt.title('Total Number of downloads')
playstore.loc[playstore.Type == 'Free', :].groupby(['Installs'])['App'].count().plot(kind = 'bar', figsize = (15,7))

_ = plt.title('Free Apps downloads')
playstore.loc[playstore.Type == 'Paid', :].groupby(['Installs'])['App'].count().plot(kind = 'bar', figsize = (15,7))

_ = plt.title('Free Apps downloads')