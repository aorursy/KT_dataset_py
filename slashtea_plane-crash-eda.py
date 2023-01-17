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
data = pd.read_csv('../input/planecrashinfo_20181121001952.csv')
data.head()
data.info()
data.replace('?', np.nan, inplace=True)
data.head()
data.isnull().any()
data.isnull().sum()
data['route'].value_counts().head()
data.drop(['registration', 'time', 'route', 'cn_ln', 'flight_no'], inplace=True, axis=1)
data.head()
data['ac_type'].value_counts().head()
data.isnull().sum()
data['summary'].replace(np.nan, ' ', inplace=True)
data.head()
data.isnull().sum()
data.dropna(inplace=True)
data.shape
data.isnull().sum()
data['date'] = pd.to_datetime(data['date'])
data.head()
data['month'] = data['date'].apply(lambda x: x.month)
data['year'] = data['date'].apply(lambda x: x.year)
data['day'] = data['date'].apply(lambda x: x.day)
data.head()
import re

data['Fatalities'] = data['fatalities'].apply(lambda x: re.search(r'[0-9]+', x).group(0))
data.head()
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [10, 5]
sns.set()
sns.barplot(data['month'].value_counts().index,
            data['month'].value_counts().get_values(), data=data)
plt.title("Number of plane Crashs")
plt.xlabel("Months")
plt.ylabel("Number of plane crashs")
sns.barplot(data['day'].value_counts().index,
            data['day'].value_counts().get_values(), data=data)
plt.title("Plane crashs by day")
plt.xlabel("days")
plt.ylabel("Plane crashs")
data['Fatalities'] = data['Fatalities'].astype('int64')
fatalities_by_month = data.groupby('month')['Fatalities'].mean()

sns.barplot(fatalities_by_month.index, fatalities_by_month.get_values())
plt.title('Mean fatalities by month')
plt.xlabel('Month')
plt.ylabel('Mean of fatalities')
