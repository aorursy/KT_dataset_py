# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Train  = pd.read_csv("../input/advertsuccess/Train.csv")
Train.head()
Train.info()
Train.isnull().sum()
plt.figure(figsize=(15,6))

sns.countplot(x='genre',data=Train,order=Train['genre'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))

sns.countplot(x='targeted_sex',data=Train,order=Train['targeted_sex'].value_counts().sort_values().index);


plt.figure(figsize=(15,6))

sns.countplot(x='industry',data=Train,order=Train['industry'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))

sns.countplot(x='realtionship_status',data=Train,order=Train['realtionship_status'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))

sns.countplot(x='genre',hue='netgain', data=Train,palette="Set1")

plt.xlabel('Genre')

plt.ylabel('Count')

plt.title('Genre Wise Netgain')

plt.show()
plt.figure(figsize=(15,6))

sns.countplot(x='realtionship_status',hue='netgain', data=Train,palette="Set1",order=Train['realtionship_status'].value_counts().sort_values().index)

plt.xlabel('Realtionship Status')

plt.ylabel('Count')

plt.title('Realtionship Status Wise Netgain')

plt.show()
plt.figure(figsize=(15,6))

sns.countplot(x='industry',hue='netgain', data=Train,palette="Set1",order=Train['industry'].value_counts().sort_values().index)

plt.xlabel('industry')

plt.ylabel('Count')

plt.title('industry Wise Netgain')

plt.show()
plt.figure(figsize=(15,6))

sns.countplot(x='targeted_sex',hue='netgain', data=Train,palette="Set1",order=Train['targeted_sex'].value_counts().sort_values().index)

plt.xlabel('Targeted sex')

plt.ylabel('Count')

plt.title('Targeted sex Wise Netgain')

plt.show()



plt.rcParams['figure.figsize'] = (18, 7)

sns.violinplot(Train['targeted_sex'], Train['netgain'], palette = 'rainbow')

plt.title('Gender vs Netgain Score', fontsize = 20)

plt.show()
plt.figure(figsize=(15,6))

sns.countplot(x='expensive',hue='netgain', data=Train,palette="Set1",order=Train['expensive'].value_counts().sort_values().index)

plt.xlabel('Expensive')

plt.ylabel('Count')

plt.title('Expensive Wise Netgain')

plt.show()
plt.figure(figsize=(15,6))

sns.distplot(Train['average_runtime(minutes_per_week)'])

plt.show()


fig, ax = plt.subplots()

ax.scatter(x = Train['ratings'], y = Train['average_runtime(minutes_per_week)'])

plt.ylabel('Average Runtime', fontsize=13)

plt.xlabel('Ratings', fontsize=13)

plt.show()