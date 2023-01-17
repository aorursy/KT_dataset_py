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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

pd.options.display.max_columns = 200

print(os.listdir("../input"))

data = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1")

data.columns

data.dtypes                          

np.iinfo('uint16')
data.info
data.describe
data.shape
data.head
data.tail
#District Wise Crime

plt.figure(figsize=(16,8))

data['DISTRICT'].value_counts().plot.bar()

plt.title('BOSTON: District wise Crimes')

plt.ylabel('Number of Crimes')

plt.xlabel('District')

plt.show()
#year wise crime trend

plt.figure(figsize=(16,8))

data['YEAR'].value_counts().plot.bar()

plt.title('BOSTON: Crimes - Yearly trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Year')

plt.show()
#hour wise crime trend

plt.figure(figsize=(16,8))

data['HOUR'].value_counts().plot.bar()

plt.title('BOSTON: Crimes - Hourly trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Hour')

plt.show()
#Weekly crime trend

plt.figure(figsize=(16,8))

data['DAY_OF_WEEK'].value_counts().plot.bar()

plt.title('BOSTON: Crimes - Weekly trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Week of Day')

plt.show()
import seaborn as sns
labels = data['YEAR'].astype('category').cat.categories.tolist()

counts = data['YEAR'].value_counts()

sizes = [counts[var_cat] for var_cat in labels]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels,  autopct='%1.1f%%',shadow=True) 

ax1.axis('equal')

plt.show()

labels = data['HOUR'].astype('category').cat.categories.tolist()

counts = data['HOUR'].value_counts()

sizes = [counts[var_cat] for var_cat in labels]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels,  autopct='%1.1f%%',shadow=True) 

ax1.axis('equal')

plt.show()
sns.pairplot(data,hue ='UCR_PART') 
sns.kdeplot(data['MONTH'], data['YEAR'] )