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
df = pd.read_excel("/kaggle/input/CustomerData.xlsx")
print("The Shape of our Dataset is",df.shape)

print(df.shape[0], "Rows")

print(df.shape[1], "Columns")
df.dtypes
df.isnull().sum()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize= (12, 5))

sns.countplot(x = "customer", data = df)

plt.title("Count of Customer Type")

plt.show()



plt.figure(figsize= (12, 5))

sns.countplot(x = "SEG", data = df)

plt.title("Distrbution of Segment")

plt.show()



plt.figure(figsize= (12, 5))

sns.countplot(x = "public_sector", data = df)

plt.title("Public Sector (1 - Yes) vs Non-Public Sector (0 - No)")

plt.show()



plt.figure(figsize= (12, 5))

sns.countplot(x = "us_region", data = df)

plt.title("Distribution of US Regions")

plt.show()



plt.figure(figsize= (18, 5))

sns.countplot(x = "STATE", data = df)

plt.title("Distribution by State")

plt.show()
print('Average Corporate Revenue', round(df['corp_rev'].mean(),2))

print('Average Revenue Last Year', round(df['rev_lastyr'].mean(),2))

print('Average Revenue This Year', round(df['rev_thisyr'].mean(),2))

print('Average Total Revenue', round(df['tot_revenue'].mean(),2))

print('Average Years Purchased', round(df['yrs_purchase'].mean(),2))

print('Average Estimated Spend', round(df['est_spend'].mean(),2))
