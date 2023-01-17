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
data = pd.read_csv("../input/train.csv")
data.head()
data.info()
data.replace('NONE', np.nan, inplace=True)
data.head()
data.isnull().sum()
data['Category'].value_counts().head()
len(data.Category.unique())
data['Dates'] = pd.to_datetime(data['Dates'])
data.info()
data['month'] = data['Dates'].apply(lambda x: x.month)
data['year'] = data['Dates'].apply(lambda x: x.year)
data['day'] = data['Dates'].apply(lambda x: x.day)

data.head()
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [10, 5]
sns.set()
sns.barplot(data['Category'].value_counts()[:5].index,
            data['Category'].value_counts()[:5].get_values(), data=data)
plt.title("Most common crime Categories")
plt.xlabel("Categories")
plt.ylabel("Frequency")
sns.barplot(data['DayOfWeek'].value_counts().index,
            data['DayOfWeek'].value_counts().get_values(), data=data)
plt.title("Which days of the week crimes happen the most")
plt.xlabel("Week days")
plt.ylabel("Occurences")
sns.barplot(data['month'].value_counts().index,
            data['month'].value_counts().get_values(), data=data)
plt.title("Crimes by month")
plt.xlabel("Months")
plt.ylabel("Total crimes")
sns.barplot(data['year'].value_counts().index,
            data['year'].value_counts().get_values(), data=data)
plt.title("Crimes by year")
plt.xlabel("Years")
plt.ylabel("Total crimes")
plt.rcParams['figure.figsize'] = [18, 6]
sns.barplot(data['day'].value_counts().index,
            data['day'].value_counts().get_values(), data=data)
plt.title("Crimes by day")
plt.xlabel("Days")
plt.ylabel("Total crimes")
