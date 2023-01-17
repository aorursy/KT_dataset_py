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


d = pd.read_csv('../input/ttl-fb/TTL_PAGE.csv')
d.head()
d.columns
d.drop(d.index[1], inplace=True)
d
d.iloc[0:,5]
d = d.rename(columns={'Daily Page Engaged Users':'Daily_Engaged_Users','28 Days Page Engaged Users':'Monthly_Engaged_Users','28 Days Total Reach':'Total_Monthly_Reach'})

d
d.Daily_Engaged_Users
d.tail()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(30,20))

plt.title("Daily organic reach over time")

sns.scatterplot(x = d['Date'],y = d['Daily Organic Reach'])
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

plt.title("Daily organic reach over a month")

sns.barplot(x = d.index,y = d['Daily Organic Reach'])

plt.xlabel("Number of days")

plt.ylabel("Organic reach")
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

plt.title("Total Monthly Reach")

sns.barplot(x = d.index,y = d['Total_Monthly_Reach'])

plt.xlabel("Number of days")

plt.ylabel("Monthly reach")
l = max(d.Monthly_Engaged_Users)

l
d.Monthly_Engaged_Users.mode()
d.describe()
d.Monthly_Engaged_Users.unique()
d.Monthly_Engaged_Users.value_counts
k = d[['Date', 'Daily_Engaged_Users', 'Monthly_Engaged_Users','Weekly Page Engaged Users','Lifetime Total Likes','Daily Total Reach','Weekly Total Reach','Total_Monthly_Reach']]

k
k.to_csv("social_media.csv")
k
k.head()
page = pd.read_csv("social_media.csv",index_col="Date", parse_dates=True)
page.head()
page
page_data = pd.read_csv("./social_media.csv")
page_data
page_data.head()
page_data.iloc[0:,0].dtype
page_data.Monthly_Engaged_Users.dtype
page_data.Daily_Engaged_Users.dtype
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

plt.title("Daily visitors in the page")

sns.barplot(x = page_data.index,y = page_data['Daily_Engaged_Users'])

plt.xlabel("Number of days")

plt.ylabel("Daily visitors in the page")
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(14,7))

plt.title("Daily visitors in the page")

sns.scatterplot(x = page_data.index,y = page_data['Daily_Engaged_Users'])

plt.xlabel("Number of days")

plt.ylabel("Daily visitors in the page")
page_data.columns
plt.figure(figsize=(14,7))

plt.title("Daily visitors in the page")

sns.swarmplot(x = page_data.index,y = page_data['Daily_Engaged_Users'])

plt.xlabel("Number of days")

plt.ylabel("Daily visitors in the page")
plt.figure(figsize=(14,7))

plt.title("Monthly visitors in the page")

sns.barplot(x = page_data.index,y = page_data['Monthly_Engaged_Users'])

plt.xlabel("Number of days")

plt.ylabel("Monthly visitors in the page")
plt.figure(figsize=(14,7))

plt.title("Monthly visitors in the page")

sns.scatterplot(x = page_data.index,y = page_data['Monthly_Engaged_Users'])

plt.xlabel("Number of days")

plt.ylabel("Monthly visitors in the page")
pd.Series([1, 2], dtype='int32')
page_data.Monthly_Engaged_Users.dtype

page_data.head()
feature = ['Date', 'Daily_Engaged_Users', 'Monthly_Engaged_Users',

       'Weekly Page Engaged Users', 'Lifetime Total Likes',

       'Daily Total Reach', 'Weekly Total Reach']
X = page_data[feature]
X.describe()
X.head()
y = page_data.Total_Monthly_Reach
y.head()
y